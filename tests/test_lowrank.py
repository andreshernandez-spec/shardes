"""LowRank, against a naive materialize-everything reference.

The property suite in test_strategies.py already covers the seed contract, chunk
additivity and linearity for every registered strategy. What is here is specific to the
factored representation: that it equals the thing it is an optimisation of, that it never
builds that thing, and that it converges to full rank as r grows.
"""

import functools
import jax
import jax.numpy as jnp
import pytest

from shardes import metrics, shaping
from shardes.estimator import estimate
from shardes.nn import dense
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.core import ShardedES
from shardes import sharding
from shardes.strategies._scale import Separable
from shardes.strategies.lowrank import LowRank, LowRankWeight

M, K, N = 6, 4, 8
RTOL = 1e-5


@pytest.fixture
def params():
    keys = jax.random.split(jax.random.key(0), 2)
    return {
        "w": jax.random.normal(keys[0], (M, K), dtype=jnp.float32),
        "scale": jax.random.normal(keys[1], (K,), dtype=jnp.float32),
    }


@functools.lru_cache(maxsize=None)
def _epsilon_fn(strategy):
    return jax.jit(lambda k, p, i, ids: strategy.contract(
        strategy.sample(k, p, ids), (ids == i).astype(jnp.float32)))


def epsilon(strategy, key, params, i, ids):
    """Member i's perturbation, via a one-hot contraction.

    Jitted and cached per strategy: eager dispatch compiles one tiny module per primitive,
    and a strategy is a few hundred of them. jit is not part of the contract under test.
    """
    return _epsilon_fn(strategy)(key, params, i, ids)


@pytest.mark.parametrize("r", [1, 2, 4])
def test_contract_matches_the_naive_loop(params, r):
    """Vectorised contract against an explicit Python loop over members.

    docs/conventions.md names this as LowRank.contract's oracle: `sum_n w_n a_n b_n^T` at
    small N, written the obvious way.
    """
    s = LowRank(r=r)
    key, ids = jax.random.key(1), jnp.arange(N)
    w = jax.random.normal(jax.random.key(2), (N,), dtype=jnp.float32)
    pert = s.sample(key, params, ids)

    got = s.contract(pert, w)

    naive = jnp.zeros((M, K))
    for n in range(N):
        a = pert.factors["w"].a[n]
        b = pert.factors["w"].b[n]
        naive = naive + w[n] * (a @ b.T) / jnp.sqrt(float(r))
    assert float(metrics.relative_mse(got["w"], naive)) < 1e-10


@pytest.mark.parametrize("r", [1, 2])
def test_never_materializes_the_perturbation(params, r):
    """Invariant 3. No (n_members, m, k) array anywhere in the jaxpr of `apply`.

    Traced rather than profiled, so it fails in CI rather than on a rented GPU. The
    banned shape is the full per-member weight; A and B carry a members axis legitimately
    and are (n, m, r) and (n, k, r), which is why `r` is excluded rather than the axis.
    """
    s = LowRank(r=r)
    ids = jnp.arange(N)
    pert = s.sample(jax.random.key(3), params, ids)
    x = jax.random.normal(jax.random.key(4), (3, K), dtype=jnp.float32)

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"])) + jnp.sum(p["scale"])

    jaxpr = jax.make_jaxpr(lambda: s.apply(model, params, pert, 0.1)(x))()
    banned = (N, M, K)
    shapes = [
        v.aval.shape
        for eqn in jaxpr.jaxpr.eqns
        for v in eqn.outvars
        if hasattr(v, "aval") and hasattr(v.aval, "shape")
    ]
    assert banned not in shapes, f"materialized {banned}: {sorted(set(shapes))}"


def test_apply_matches_an_explicitly_perturbed_model(params):
    """The factored forward pass against forming W + sigma*E per member and evaluating.

    This is the equivalence the whole strategy is an optimisation of. If it fails, the
    speedup is measuring a different computation.
    """
    r, sigma = 2, 0.1
    s = LowRank(r=r)
    ids = jnp.arange(N)
    pert = s.sample(jax.random.key(5), params, ids)
    x = jax.random.normal(jax.random.key(6), (3, K), dtype=jnp.float32)

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"])) + jnp.sum(p["scale"])

    got = s.apply(model, params, pert, sigma)(x)

    want = []
    for n in range(N):
        f = pert.factors
        e = (f["w"].a[n] @ f["w"].b[n].T) / jnp.sqrt(float(r))
        explicit = {"w": params["w"] + sigma * e, "scale": params["scale"] + sigma * f["scale"].a[n]}
        want.append(model(explicit, x))
    assert jnp.allclose(got, jnp.stack(want), rtol=1e-4)


def test_dense_leaves_are_perturbed_densely(params):
    """A vector has no two-factor decomposition, so `LowRank` must fall back rather than
    invent one. Shared with dimensions.sampling_dimension, which counts it the same way."""
    s = LowRank(r=1)
    pert = s.sample(jax.random.key(7), params, jnp.arange(N))
    assert pert.factors["scale"].b is None
    assert pert.factors["scale"].a.shape == (N, K)
    assert pert.factors["w"].b is not None


def test_degenerates_to_iid_gaussian_on_a_vector_only_tree():
    """Every leaf dense means LowRank *is* IIDGaussian. Not a coincidence worth hiding:
    it is why the estimator's unbiasedness tests cover LowRank at all."""
    flat = {"theta": jnp.zeros((16,), dtype=jnp.float32)}
    key, ids = jax.random.key(8), jnp.arange(5)
    w = jax.random.normal(jax.random.key(9), (5,), dtype=jnp.float32)

    lr = LowRank(r=1).contract(LowRank(r=1).sample(key, flat, ids), w)
    iid = IIDGaussian().contract(IIDGaussian().sample(key, flat, ids), w)
    assert float(metrics.relative_mse(lr, iid)) < 1e-10


def test_a_model_that_bypasses_dense_fails_loudly(params):
    """A model doing arithmetic on a weight instead of calling `dense` cannot be reached
    by a structured perturbation. It must raise, not quietly compute something else.

    This is the cost of the deferred arbitrary-model decision (docs/01 C0.1), made
    visible: `LowRank` requires the model to go through the seam.
    """
    s = LowRank(r=1)
    pert = s.sample(jax.random.key(12), params, jnp.arange(3))
    bypassing = lambda p, _b: jnp.sum(p["w"] * 2.0)
    with pytest.raises(TypeError):
        s.apply(bypassing, params, pert, 0.1)(None)


@pytest.mark.slow
def test_estimator_converges_to_full_rank_as_r_grows():
    """docs/01: LowRank(r) estimator -> IIDGaussian estimator as r grows.

    The claim is about the *estimator*, not about a single contraction. Two contractions
    from the same key are independent draws whose gap is ~2x the variance at any r, so
    comparing them measures nothing; what converges is estimate quality against a known
    gradient.

    Measured at m=12, k=10, n=64, sigma=0.02, 128 replicates:
    r=1 0.5326, r=2 0.5578, r=4 0.5676, r=10 0.5809, full 0.5822. At r = min(m, k) the
    factorisation is no longer a restriction, which is why r=10 lands on full rank.
    """
    m, k, b, n = 12, 10, 16, 64
    keys = jax.random.split(jax.random.key(0), 3)
    batch = (
        jax.random.normal(keys[0], (b, k), dtype=jnp.float32),
        jax.random.normal(keys[1], (b, m), dtype=jnp.float32),
    )
    p = {"w": jax.random.normal(keys[2], (m, k), dtype=jnp.float32)}

    def model(pp, bb):
        return 0.5 * jnp.mean(jnp.square(dense(bb[0], pp["w"]) - bb[1]))

    truth = jax.grad(model)(p, batch)
    ids = jnp.arange(n)

    def quality(strategy):
        gs = jax.vmap(
            lambda kk: estimate(strategy, model, p, batch, kk, member_ids=ids,
                                sigma=0.02, shaping=shaping.centered)
        )(jax.random.split(jax.random.key(3), 128))
        return float(jnp.mean(jax.vmap(lambda g: metrics.cosine_similarity(g, truth))(gs)))

    cos = [quality(LowRank(r=r)) for r in (1, 2, 4, 10)]
    full = quality(IIDGaussian())

    assert cos == sorted(cos), f"not monotone in r: {cos}"
    assert cos[0] < full, "rank 1 should be strictly worse than full rank"
    assert abs(cos[-1] - full) < 0.01, f"r=min(m,k) should reach full rank: {cos[-1]} vs {full}"


def test_low_rank_weight_is_a_structured_weight():
    from shardes.nn import StructuredWeight

    w = LowRankWeight(jnp.zeros((M, K)), jnp.zeros((M, 1)), jnp.zeros((K, 1)), jnp.float32(1.0))
    assert isinstance(w, StructuredWeight)
    assert len(jax.tree.leaves(w)) == 4


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_a_nonsense_rank(bad):
    with pytest.raises(ValueError, match="rank must be at least 1"):
        LowRank(r=bad)


def test_a_structured_weight_is_not_a_sequence():
    """Regression: `LowRankWeight` was a NamedTuple and inherited tuple semantics, so a model
    writing the natural thing got a **wrong answer silently** rather than an error.

        table[0]          returned the base matrix `w`, not row 0
        len(table)        returned 4, the field count, not the vocabulary size
        for row in table  iterated the four fields

    None of those raised. That is the one failure mode worse than the seam constraint itself,
    because the seam constraint is at least loud. A registered dataclass is not a sequence, so
    all three are refused, and the message names `dense` or `embed`.

    Being a pytree is unaffected, which is the property that made NamedTuple attractive: four
    leaves either way. `test_low_rank_weight_is_a_structured_weight` pins that.
    """
    w = LowRankWeight(jnp.arange(24, dtype=jnp.float32).reshape(6, 4),
                      jnp.ones((6, 1), jnp.float32), jnp.ones((4, 1), jnp.float32),
                      jnp.float32(0.1))
    assert not isinstance(w, tuple)
    for label, call in (
        ("indexing", lambda: w[0]),
        ("len", lambda: len(w)),
        ("iteration", lambda: list(w)),
        ("transposition", lambda: w.T),
        ("elementwise", lambda: w * 2.0),
        ("matmul", lambda: w @ jnp.ones((4, 2))),
    ):
        with pytest.raises(TypeError, match="shardes.nn"):
            call()
    # metadata stays available: a caller's own shape assertions should not have to change
    assert w.shape == (6, 4) and w.dtype == jnp.float32


# --------------------------------------------------------------------------------------
# Separable sigma: the per-coordinate family a factored perturbation can carry.
# docs/proposal-review-fixes.md, implemented 2026-08-11.
# --------------------------------------------------------------------------------------


def test_a_separable_sigma_equals_its_dense_form():
    """`(u v^T) * (A B^T) == (u * A) @ (v * B).T`, which is why the type exists.

    The forward pass folds `u` and `v` into the factors and never forms the `(m, k)` product.
    This asserts that shortcut against the thing it is a shortcut for: materialise
    `W + (u v^T) * (A B^T)` per member and evaluate the model on it.

    A general per-coordinate sigma has no such identity. It raises the rank of the product
    above `r`, measured 4 against `r=2`, so there is nothing left to apply as two GEMMs and
    `apply` refuses it.
    """
    mesh = sharding.make_mesh(1)
    p0 = {"w": jax.random.normal(jax.random.key(3), (6, 4))}
    x = jax.random.normal(jax.random.key(4), (2, 4))
    u = jnp.abs(jax.random.normal(jax.random.key(1), (6,))) + 0.5
    v = jnp.abs(jax.random.normal(jax.random.key(2), (4,))) + 0.5

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"]) ** 2)

    es = ShardedES(LowRank(r=1), n=4, sigma={"w": Separable(u, v)}, lr=0.05, mesh=mesh,
                   how="A")
    state = es.init(jax.random.key(0), p0)
    pert, state = es.ask(state)
    got = es.apply(model, state, pert)(x)

    factors = pert.factors["w"]

    def materialised(a, b):
        w = p0["w"] + jnp.outer(u, v) * (a @ b.T)
        return jnp.sum((x @ w.T) ** 2)

    assert jnp.allclose(got, jax.vmap(materialised)(factors.a, factors.b), rtol=1e-4)


def test_a_uniform_separable_equals_a_scalar_sigma():
    """The widening has to be a superset, which is the promise the general diagonal broke."""
    mesh = sharding.make_mesh(1)
    p0 = {"w": jax.random.normal(jax.random.key(3), (6, 4))}
    x = jax.random.normal(jax.random.key(4), (2, 4))
    s = 0.02

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"]) ** 2)

    def run(sigma):
        es = ShardedES(LowRank(r=1), n=4, sigma=sigma, lr=0.05, mesh=mesh, how="A")
        state = es.init(jax.random.key(0), p0)
        pert, state = es.ask(state)
        return es.apply(model, state, pert)(x)

    separable = {"w": Separable(jnp.full((6,), s), jnp.ones((4,)))}
    assert jnp.allclose(run(separable), run(s), rtol=1e-6)


def test_a_separable_never_materialises_the_perturbation():
    """Invariant 3 still holds with a per-coordinate sigma, which is the entire point.

    Folding `u` and `v` into `(m, r)` and `(k, r)` keeps every array factor-sized. A version
    that densified the sigma and multiplied would satisfy every other test here and quietly
    allocate `(n, m, k)`.
    """
    mesh = sharding.make_mesh(1)
    p0 = {"w": jnp.ones((6, 4))}
    u, v = jnp.full((6,), 0.01), jnp.full((4,), 0.02)

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"]))

    es = ShardedES(LowRank(r=1), n=4, sigma={"w": Separable(u, v)}, lr=0.05, mesh=mesh,
                   how="A")
    state = es.init(jax.random.key(0), p0)
    pert, state = es.ask(state)
    jaxpr = jax.make_jaxpr(es.apply(model, state, pert))(jnp.ones((2, 4)))
    forbidden = (4, 6, 4)
    shapes = [tuple(v_.aval.shape) for v_ in jaxpr.jaxpr.eqns[0].invars if hasattr(v_, "aval")]
    assert forbidden not in [tuple(s) for s in shapes], "materialised (n, m, k)"


def test_a_separable_on_a_densely_perturbed_leaf_is_refused():
    """A vector has one axis, so an output/input split is not a thing it has."""
    mesh = sharding.make_mesh(1)
    p0 = {"b": jnp.ones((4,))}
    es = ShardedES(LowRank(r=1), n=4, sigma={"b": Separable(jnp.ones((4,)), jnp.ones((4,)))},
                   lr=0.05, mesh=mesh, how="A")
    state = es.init(jax.random.key(0), p0)
    pert, state = es.ask(state)
    with pytest.raises(ValueError, match="perturbs densely"):
        es.apply(lambda p, x: jnp.sum(p["b"]), state, pert)(jnp.zeros(()))


def test_a_separable_with_the_wrong_shapes_names_the_leaf():
    """Caught at trace time, not several lines later inside a GEMM."""
    mesh = sharding.make_mesh(1)
    p0 = {"w": jnp.ones((6, 4))}
    es = ShardedES(LowRank(r=1), n=4, sigma={"w": Separable(jnp.ones((5,)), jnp.ones((4,)))},
                   lr=0.05, mesh=mesh, how="A")
    state = es.init(jax.random.key(0), p0)
    pert, state = es.ask(state)
    with pytest.raises(ValueError, match=r"u of shape \(6,\)"):
        es.apply(lambda p, x: jnp.sum(dense(x, p["w"])), state, pert)(jnp.ones((2, 4)))
