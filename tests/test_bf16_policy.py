"""Policy A, master weights: docs/proposal-bf16-policy.md, accepted 2026-08-14.

The contract under test: the state carries params at f32 or wider, the model and every
noise derivation see a per-generation view in `compute_dtype`, and only the SGD step
touches the master. The failure modes these tests pin are the ones the proposal measured:
silent promotion (bf16 in, f32 model from generation 1), deleted updates (a bf16 carry
rounds fine-tuning steps away), and a scan that cannot carry the state.

The test model routes its matmul through `shardes.nn.dense`, and not only from taste: a
first draft read `params["w"] ** 2` directly and `LowRank` refused it at trace time, which
is `shardes.nn`'s structured-weight contract doing its job against this file too.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import sharding
from shardes.core import ShardedES
from shardes.nn import dense
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

STRATEGIES = [
    ("iid_gaussian", IIDGaussian),
    ("seed_regenerated", SeedRegenerated),
    ("mirrored_seed", lambda: Mirrored(SeedRegenerated())),
    ("mirrored_lr1", lambda: Mirrored(LowRank(r=1))),
]
IDS = [name for name, _ in STRATEGIES]
MAKES = [make for _, make in STRATEGIES]

D = 8  # model width; parameters are one (D, D) matrix and one (D,) vector


def _params(dtype=jnp.bfloat16):
    # 0.05 is the transformer block's init scale at d=512; the tests should live at the
    # magnitudes the library runs at, not at 1.0 where bf16 is at its kindest.
    k1, k2 = jax.random.split(jax.random.key(0))
    return {
        "w": (0.05 * jax.random.normal(k1, (D, D), jnp.float32)).astype(dtype),
        "b": (0.05 * jax.random.normal(k2, (D,), jnp.float32)).astype(dtype),
    }


def _es(make, *, n=8, compute_dtype=jnp.bfloat16, devices=1, **kw):
    return ShardedES(
        make(), n=n, sigma=0.01, lr=0.05, mesh=sharding.make_mesh(devices),
        compute_dtype=compute_dtype, **kw,
    )


_X0 = jnp.ones((D,), jnp.bfloat16)


def _loss(params, x):
    """Quadratic in every parameter, read only through the seams."""
    h = dense(jnp.eye(D, dtype=x.dtype), params["w"])
    return jnp.sum(h**2) + jnp.sum(params["b"] ** 2) + 0.0 * jnp.sum(x)


def _linear(params, x):
    """Linear fitness with a constant, known gradient: dL/dW = 1 x0^T, dL/db = 1."""
    return jnp.sum(dense(_X0[None, :], params["w"])) + jnp.sum(params["b"]) + 0.0 * jnp.sum(x)


def test_bf16_params_without_a_policy_are_refused():
    """The silent alternative is what the code used to do: promote after one tell."""
    es = ShardedES(IIDGaussian(), n=8, sigma=0.01, lr=0.05, mesh=sharding.make_mesh(1))
    with pytest.raises(ValueError, match="compute_dtype"):
        es.init(jax.random.key(0), _params(jnp.bfloat16))


def test_f32_params_with_no_policy_are_untouched():
    """The default path must be byte-identical to what it always was."""
    es = ShardedES(IIDGaussian(), n=8, sigma=0.01, lr=0.05, mesh=sharding.make_mesh(1))
    p = _params(jnp.float32)
    st = es.init(jax.random.key(0), p)
    assert st.params["w"] is p["w"], "f32 params were copied or cast on the no-policy path"


@pytest.mark.parametrize("make", MAKES, ids=IDS)
def test_the_model_runs_in_the_compute_dtype(make):
    """Trace-time assert inside the model: every leaf it sees carries bf16.

    This is the test that fails without the sigma cast in the strategies: an f32 sigma
    promotes `p + s * e` back to f32 through JAX promotion and the model quietly runs in
    the master's dtype. `LowRankWeight` exposes `dtype` for exactly this kind of check.
    """
    es = _es(make)
    st = es.init(jax.random.key(0), _params(jnp.bfloat16))
    pert, st = es.ask(st)

    def model(params, x):
        assert params["w"].dtype == jnp.bfloat16, params["w"].dtype
        assert params["b"].dtype == jnp.bfloat16, params["b"].dtype
        return _loss(params, x)

    fit = es.apply(model, st, pert)(jnp.zeros((), jnp.bfloat16))
    assert fit.shape == (8,)


@pytest.mark.parametrize("make", MAKES, ids=IDS)
def test_the_master_stays_f32_through_tell(make):
    es = _es(make)
    st = es.init(jax.random.key(0), _params(jnp.bfloat16))
    assert st.params["w"].dtype == jnp.float32
    pert, st = es.ask(st)
    fit = es.apply(_loss, st, pert)(jnp.zeros((), jnp.bfloat16))
    st = es.tell(st, pert, fit)
    assert st.params["w"].dtype == jnp.float32
    assert st.params["b"].dtype == jnp.float32


def test_a_generation_scan_carries_the_state():
    """The forcing function from the proposal: this failed outright before, because tell
    promoted the carry from bf16 to f32 and scan rejects a carry that changes type."""
    es = _es(lambda: Mirrored(SeedRegenerated()))
    st = es.init(jax.random.key(0), _params(jnp.bfloat16))

    def gen(state, _):
        pert, state = es.ask(state)
        fit = es.apply(_loss, state, pert)(jnp.zeros((), jnp.bfloat16))
        return es.tell(state, pert, fit), fit

    final, fits = jax.lax.scan(gen, st, None, length=3)
    assert final.params["w"].dtype == jnp.float32
    assert int(final.generation) == 3
    assert fits.shape == (3, 8)


def test_f32_input_with_bf16_compute_is_allowed():
    """An f32 checkpoint evaluated in bf16 is a legitimate configuration, not an error."""
    es = _es(IIDGaussian)
    st = es.init(jax.random.key(0), _params(jnp.float32))
    assert st.params["w"].dtype == jnp.float32
    pert, st = es.ask(st)

    def model(params, x):
        assert params["w"].dtype == jnp.bfloat16
        return _loss(params, x)

    es.apply(model, st, pert)(jnp.zeros((), jnp.bfloat16))


@pytest.mark.parametrize("make", MAKES, ids=IDS)
def test_forward_and_contraction_use_the_same_noise(make):
    """The estimator works end to end in bf16, which pins the one bug this policy could
    reintroduce silently: `ask`/`apply` sampling noise in one dtype and `tell`
    regenerating it in another. The noise is a deterministic function of (key, id, dtype),
    so a dtype mismatch makes the contraction correlate fitnesses with perturbations that
    were never evaluated, and the update decorrelates from the gradient entirely.

    A linear fitness makes that measurable with no convergence question: `f_i` is exactly
    `<grad, eps_i>` plus a constant, so the contracted update must align with the
    gradient. Measured alignment is 0.65 to 0.79 across the four strategies at this
    shape; a mismatch reads about zero. The threshold is far below the measured band and
    far above the failure.
    """
    es = _es(make, n=128, shaping=lambda f: f)
    st = es.init(jax.random.key(0), _params(jnp.bfloat16))
    pert, st = es.ask(st)
    fit = es.apply(_linear, st, pert)(jnp.zeros((), jnp.bfloat16))
    st2 = es.tell(st, pert, fit)

    update = jnp.concatenate(
        [(a - b).ravel() for a, b in zip(jax.tree.leaves(st2.params),
                                         jax.tree.leaves(st.params))]
    )
    # Leaves in dict order: b then w. tell descends, so the update opposes the gradient.
    grad = jnp.concatenate([
        jnp.ones((D,), jnp.float32).ravel(),
        (jnp.ones((D, 1)) @ _X0[None, :].astype(jnp.float32)).ravel(),
    ])
    cos = float(jnp.vdot(update, -grad) / (jnp.linalg.norm(update) * jnp.linalg.norm(grad)))
    assert cos > 0.4, f"update/gradient alignment {cos:.3f}; dtype-mismatched noise?"


@pytest.mark.parametrize("make", MAKES, ids=IDS)
@pytest.mark.parametrize("how", ["A", "B"])
def test_device_invariance_holds_under_bf16_compute(make, how):
    """Invariant 2 at the compute dtype the library will actually ship at scale.

    The tolerance matches the f32 invariance test (1e-6), not the 1e-2 that
    docs/conventions.md reserves for bf16 accumulation paths, and that is the point of
    policy A: the noise is quantized to bf16 identically on every device count (same
    keys, same dtype), the accumulation is f32 on every path, so the only D-dependence
    is reassociation of the f32 sums, exactly as in the f32 test.
    """
    def run(devices):
        es = _es(make, n=16, devices=devices, how=how)
        st = es.init(jax.random.key(3), _params(jnp.bfloat16))
        pert, st = es.ask(st)
        fit = es.apply(_loss, st, pert)(jnp.zeros((), jnp.bfloat16))
        return jax.device_get(es.tell(st, pert, fit).params)

    ref, got = run(1), run(8)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        err = float(abs(a - b).max() / (abs(a).max() + 1e-30))
        assert err < 1e-6, f"{how}: relative error {err:.2e} across device counts"
