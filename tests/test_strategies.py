"""Properties every PerturbationStrategy must satisfy, whatever its scheme.

Two halves.

`test_protocol_*` guard the agreed signature itself (docs/01 C0.1) and run today. The
signature is load-bearing enough that drifting it silently would be worse than a
compile error, and Protocols give no compile error.

The parametrized suite covers every strategy in `shardes.strategies.registry.STRATEGIES`.
That registry is empty until the first strategy lands, so those tests skip rather than
pass vacuously. Register a strategy there and everything below applies to it immediately,
with no edit here.

Observation channel: `Perturbation` is opaque by design, so member i is read out by
contracting with a one-hot weight vector. That keeps these tests independent of any
strategy's internal layout, which is the whole point of the protocol being structural.
"""

import inspect

import functools
import jax
import jax.numpy as jnp
import pytest

from shardes.nn import dense
from shardes.strategies.protocol import Perturbation, PerturbationStrategy
from shardes.strategies.registry import REPRESENTATIVES, STRATEGIES

RTOL = 1e-6


def rel_err(got, want):
    flat_g = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(got)])
    flat_w = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(want)])
    return jnp.linalg.norm(flat_g - flat_w) / jnp.linalg.norm(flat_w)


@functools.lru_cache(maxsize=None)
def _contract_fn(strategy):
    """The jitted sample-and-contract for one strategy, built once.

    **Cached on the strategy, and that is the whole point.** `jax.jit` keys its compilation
    cache on the function object it was handed. Building `jax.jit(lambda ...)` inside the
    helper made a new lambda per call, so every call was a cache miss and recompiled the
    whole strategy: measured 2.3 s per call for `mirrored_hd_lr4`, against 2.3 s once and
    then 0.3 ms. A test that contracts under five different weight vectors paid five
    compiles for one program.

    The previous version's docstring said "one compile instead of hundreds", and it did
    replace per-primitive dispatch, so the fast tier really did drop from 98 s. It was one
    compile per *call*, though, which is what this fixes.
    """
    return jax.jit(lambda k, p, i, w: strategy.contract(strategy.sample(k, p, i), w))


def contracted(strategy, base_key, params, member_ids, weights):
    """sample then contract, under one jit.

    `jit` here is about the suite's runtime, not about what is being tested. A strategy is a
    few hundred primitives (an FWHT chain, a 30-step XOR, a scan), and dispatching each one
    eagerly compiles a tiny HLO module per primitive. Nothing about the strategies changes;
    jit is not part of the contract.
    """
    return _contract_fn(strategy)(base_key, params, member_ids, weights)


def epsilon(strategy, base_key, params, member_id, member_ids):
    """Member `member_id`'s perturbation, read out via a one-hot contraction.

    `member_ids` is the batch it was sampled as part of, which is exactly the thing the
    seed contract says must not matter.
    """
    onehot = (member_ids == member_id).astype(jnp.float32)
    return contracted(strategy, base_key, params, member_ids, onehot)


# --------------------------------------------------------------------------------------
# Signature guards. These run now.
# --------------------------------------------------------------------------------------


def test_protocol_signature_is_the_agreed_one():
    """docs/01 C0.1. Parameter names are part of the contract, not decoration."""
    expected = {
        "sample": ["self", "base_key", "params", "member_ids"],
        "apply": ["self", "model", "params", "pert", "sigma"],
        "contract": ["self", "pert", "weights"],
    }
    for name, want in expected.items():
        got = list(inspect.signature(getattr(PerturbationStrategy, name)).parameters)
        assert got == want, f"{name}{tuple(got)} drifted from {tuple(want)}"


def test_protocol_has_no_extra_methods():
    """Four steps. **The fourth is the abstraction leaking, and it was accepted knowingly.**

    This said three, with "a fourth would mean the abstraction leaked", and that reading was
    right: `split` is the protocol admitting it knows about devices, which
    `sharding.AXIS_TYPE_NOTE` spent its argument avoiding. The note said to revisit "if the
    strategy protocol ever grows a sharding-aware seam for another reason", and this is the
    reason, measured:

    `ShardedES.apply` divides the population by vmapping device rows, and it cannot reshape
    the perturbation itself because only a strategy knows which of its arrays carry a member
    axis. `SeedPerturbation.like` is the params tree and carries none;
    `MirroredPerturbation.inner` carries `n/2`. The alternative was re-deriving the
    perturbation per row, which is free for `SeedRegenerated` and **1.32x the per-device
    FLOPs** for `IIDGaussian`, whose `sample` is the work. With `split` that is 1.108x, and
    the other three strategies return to their pre-reshape cost exactly.

    `docs/proposal-scan-strategies-distribute.md` carries the four options and the numbers.

    The guard stays, at four. A fifth method wants the same argument made again.
    """
    methods = {n for n in vars(PerturbationStrategy) if not n.startswith("_")}
    assert methods == {"sample", "apply", "contract", "split"}


def test_sample_cannot_take_sigma():
    """Unit scale is structural: there is nowhere in `sample` to put a sigma."""
    assert "sigma" not in inspect.signature(PerturbationStrategy.sample).parameters


def test_contract_cannot_take_sigma():
    """The 1/(N sigma) factor belongs to tell, so contract must not see sigma."""
    assert "sigma" not in inspect.signature(PerturbationStrategy.contract).parameters


def test_perturbation_carries_regeneration_state():
    """Without these two fields, Strategy A cannot re-derive and the type splits in two."""
    assert set(Perturbation.__annotations__) >= {"base_key", "member_ids"}


# --------------------------------------------------------------------------------------
# Properties. Skipped until STRATEGIES is populated.
# --------------------------------------------------------------------------------------

_reason = "no strategy registered yet; see src/shardes/strategies/registry.py"

# The `or [...]` fallback exists because an empty parametrize list gives a bare "empty
# parameter set" skip with no reason attached, which reads as "nothing to do here" rather
# than "waiting on an implementation". Delete it once the registry is populated.
parametrize = pytest.mark.parametrize(
    "strategy",
    [pytest.param(entry.build(), id=name,
                  marks=() if name in REPRESENTATIVES else pytest.mark.slow)
     for name, entry in STRATEGIES.items()]
    or [pytest.param(None, id="none", marks=pytest.mark.skip(reason=_reason))],
)


@pytest.fixture
def params():
    k1, k2 = jax.random.split(jax.random.key(0))
    return {
        "w": jax.random.normal(k1, (8, 5), dtype=jnp.float32),
        "b": jax.random.normal(k2, (5,), dtype=jnp.float32),
    }


@parametrize
def test_seed_by_member_index(strategy, params):
    """Member 7 is member 7 regardless of the batch it was drawn in.

    This is invariant 2 in CLAUDE.md, checked without any device involved: if it holds
    across batch shapes it holds across device counts, because member_ids is the only
    thing sharding changes.
    """
    key = jax.random.key(1)
    alone = epsilon(strategy, key, params, 7, jnp.array([6, 7]))
    in_batch = epsilon(strategy, key, params, 7, jnp.arange(100))
    shuffled = epsilon(strategy, key, params, 7, jnp.array([6, 7, 2, 3]))
    assert rel_err(alone, in_batch) < RTOL
    assert rel_err(alone, shuffled) < RTOL


@parametrize
def test_contract_chunks_additively(strategy, params):
    """Partial contractions over disjoint members sum to the whole.

    This one property is what makes chunking, Strategy B's psum, and streaming full rank
    at large N all work. If it fails, none of the three are safe.
    """
    key = jax.random.key(2)
    ids = jnp.arange(16)
    w = jax.random.normal(jax.random.key(3), (16,), dtype=jnp.float32)

    whole = contracted(strategy, key, params, ids, w)
    lo = contracted(strategy, key, params, ids[:6], w[:6])
    hi = contracted(strategy, key, params, ids[6:], w[6:])
    assert rel_err(jax.tree.map(jnp.add, lo, hi), whole) < RTOL


@parametrize
def test_contract_is_linear_in_weights(strategy, params):
    key, ids = jax.random.key(4), jnp.arange(12)
    w1 = jax.random.normal(jax.random.key(5), (12,), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(6), (12,), dtype=jnp.float32)
    a, b = 2.0, -0.5

    got = contracted(strategy, key, params, ids, a * w1 + b * w2)
    want = jax.tree.map(
        lambda x, y: a * x + b * y,
        contracted(strategy, key, params, ids, w1),
        contracted(strategy, key, params, ids, w2),
    )
    assert rel_err(got, want) < RTOL


@parametrize
def test_contract_preserves_pytree_structure(strategy, params):
    """Update tree matches params tree, leaf for leaf. No global flattening."""
    key, ids = jax.random.key(7), jnp.arange(4)
    out = contracted(strategy, key, params, ids, jnp.ones(4))
    assert jax.tree.structure(out) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(params)):
        assert a.shape == b.shape


@parametrize
def test_sample_is_deterministic(strategy, params):
    """Same key, same ids, same perturbation. An experiment you cannot re-run is not one."""
    key, ids = jax.random.key(8), jnp.arange(10)
    # Not uniform weights: under Mirrored the pair contributions cancel exactly, so the
    # contraction would be zero and rel_err a 0/0.
    w = jax.random.normal(jax.random.key(80), (10,), dtype=jnp.float32)
    a = contracted(strategy, key, params, ids, w)
    b = contracted(strategy, key, params, ids, w)
    assert rel_err(a, b) == 0.0


@parametrize
def test_perturbation_is_unit_scale(strategy, params):
    """Per-leaf second moment near 1. Sigma is applied by the core, never here.

    Loose tolerance on purpose: this catches a missing or doubled scale factor, not a
    subtly wrong distribution. Distributional correctness is test_estimator's job.

    Weights are random, not uniform. Uniform weights are a degenerate case: under
    Mirrored the pair contributions cancel exactly and the contraction is identically
    zero, which is correct behaviour and useless as a scale check. With w ~ N(0,1)/sqrt(n)
    the second moment is 1 for both the plain and the antithetic case.
    """
    key = jax.random.key(9)
    ids = jnp.arange(2048)
    w = jax.random.normal(jax.random.key(90), (len(ids),), dtype=jnp.float32)
    scaled = contracted(strategy, key, params, ids, w / jnp.sqrt(len(ids)))
    for leaf in jax.tree.leaves(scaled):
        assert 0.5 < float(jnp.sqrt(jnp.mean(leaf**2))) < 2.0


@parametrize
def test_apply_evaluates_each_member_at_its_own_epsilon(strategy, params):
    """g(x) returns one output per member, perturbed by sigma times *that* member's eps.

    The model sums its params, so the expected value per member is computable from
    `contract` alone and the test never has to look inside the perturbation.

    The model goes through `shardes.nn.dense`, which is required rather than stylistic:
    a structured strategy substitutes a weight that is not an array, so a model doing
    arithmetic on it directly raises. That was predicted when this test was written and
    LowRank duly failed it.

    `x` is all ones, so `sum(dense(x, W))` is `sum(W)` and the expected value stays
    computable from `contract` alone, without the test knowing any strategy's layout.
    """
    sigma, ids = 0.1, jnp.arange(6)
    x = jnp.ones((params["w"].shape[-1],), dtype=jnp.float32)

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"])) + jnp.sum(p["b"])

    pert = strategy.sample(jax.random.key(11), params, ids)
    got = strategy.apply(model, params, pert, sigma)(x)
    assert got.shape == (6,)

    base = sum(float(jnp.sum(leaf)) for leaf in jax.tree.leaves(params))
    for i in range(6):
        eps_i = strategy.contract(pert, (ids == i).astype(jnp.float32))
        want = base + sigma * sum(float(jnp.sum(leaf)) for leaf in jax.tree.leaves(eps_i))
        assert jnp.isclose(got[i], want, rtol=1e-4), f"member {i}"


@parametrize
def test_apply_scales_with_sigma(strategy, params):
    """Doubling sigma doubles the deviation from the unperturbed model."""
    ids = jnp.arange(4)
    x = jnp.ones((params["w"].shape[-1],), dtype=jnp.float32)

    def model(p, xx):
        return jnp.sum(dense(xx, p["w"])) + jnp.sum(p["b"])

    pert = strategy.sample(jax.random.key(12), params, ids)
    base = model(params, x)
    one = strategy.apply(model, params, pert, 0.1)(x) - base
    two = strategy.apply(model, params, pert, 0.2)(x) - base
    assert jnp.allclose(two, 2 * one, rtol=1e-4)


def test_seed_regenerated_matches_iid_gaussian(params):
    """Qiu's seed trick has to reproduce full-rank noise exactly, or it is a different
    algorithm rather than a different schedule.

    The two share `_noise.member_noise`, so this is really a guard that they stay shared:
    if someone reimplements the derivation in one of them, this fires.
    """
    from shardes.strategies.iid_gaussian import IIDGaussian
    from shardes.strategies.seed_regenerated import SeedRegenerated

    key, ids = jax.random.key(21), jnp.arange(12)
    w = jax.random.normal(jax.random.key(22), (12,), dtype=jnp.float32)

    a, b = IIDGaussian(), SeedRegenerated()
    got = contracted(a, key, params, ids, w)
    want = contracted(b, key, params, ids, w)
    assert rel_err(got, want) < RTOL


@parametrize
def test_strategy_conforms_to_the_protocol(strategy):
    """Structural check. runtime_checkable only verifies the methods exist, which is
    exactly the failure the signature guards above cannot see: a strategy that forgot
    `contract` entirely."""
    assert isinstance(strategy, PerturbationStrategy)


@parametrize
def test_perturbation_conforms_to_the_protocol(strategy, params):
    pert = strategy.sample(jax.random.key(10), params, jnp.arange(4))
    assert hasattr(pert, "base_key") and hasattr(pert, "member_ids")
    assert jnp.array_equal(pert.member_ids, jnp.arange(4))


def test_chunked_seed_regenerated_matches_sequential():
    """chunk is a bandwidth optimization, not an algorithm: same members, same noise,
    same outputs up to vmap-vs-scan reassociation. Held tight because if chunking
    changed results the whole E13 wall-clock story would be buying different science."""
    import jax
    import jax.numpy as jnp

    from shardes import sharding
    from shardes.core import ShardedES
    from shardes.strategies.mirrored import Mirrored
    from shardes.strategies.seed_regenerated import SeedRegenerated

    def run(chunk):
        es = ShardedES(Mirrored(SeedRegenerated(chunk=chunk)), n=8, sigma=0.01, lr=0.05,
                       mesh=sharding.make_mesh(1))
        params = {"w": 0.05 * jax.random.normal(jax.random.key(0), (6, 6))}
        st = es.init(jax.random.key(1), params)
        pert, st = es.ask(st)
        fit = es.apply(lambda p, x: jnp.sum(p["w"] ** 2) + 0.0 * x, st, pert)(jnp.zeros(()))
        return fit, es.tell(st, pert, fit).params["w"]

    (f1, w1), (f4, w4) = run(1), run(4)
    assert jnp.allclose(f1, f4, rtol=1e-6), "chunked fitness diverged from sequential"
    assert jnp.allclose(w1, w4, rtol=1e-6), "chunked update diverged from sequential"


def test_chunk_must_divide_the_per_device_population():
    import jax
    import jax.numpy as jnp
    import pytest

    from shardes import sharding
    from shardes.core import ShardedES
    from shardes.strategies.seed_regenerated import SeedRegenerated

    es = ShardedES(SeedRegenerated(chunk=3), n=8, sigma=0.01, lr=0.05,
                   mesh=sharding.make_mesh(1))
    st = es.init(jax.random.key(0), {"w": jnp.ones((4, 4))})
    pert, st = es.ask(st)
    with pytest.raises(ValueError, match="chunk"):
        es.apply(lambda p, x: jnp.sum(p["w"]) + 0.0 * x, st, pert)(jnp.zeros(()))


def test_lowrank_sample_column_seed_contract():
    """Pins the per-column seed layout: column j of member i's A factor is
    coupling(cols[j], i, m) with cols = split(leaf stream, 2r), and the b side offset
    by r. tell regenerates factors from seeds through exactly this map (contraction
    re-runs sample), so any implementation of sample must reproduce it bit for bit;
    drift here is a different update, not a tolerance question."""
    from shardes.coupling import GAUSSIAN
    from shardes.strategies._noise import leaf_streams
    from shardes.strategies.lowrank import LowRank

    r, m, k = 5, 7, 6
    params = {"w": jnp.zeros((m, k), jnp.float32)}
    pert = LowRank(r=r).sample(jax.random.key(3), params, jnp.arange(2))
    (stream,) = leaf_streams(jax.random.key(3), 1)
    cols = jax.random.split(stream, 2 * r)
    lf = pert.factors["w"]
    for i in range(2):
        member = jnp.int32(i)
        for j in range(r):
            assert jnp.array_equal(lf.a[i, :, j], GAUSSIAN(cols[j], member, m, jnp.float32))
            assert jnp.array_equal(lf.b[i, :, j], GAUSSIAN(cols[r + j], member, k, jnp.float32))


def test_lowrank_sample_graph_is_rank_independent():
    """sample sits inside tell's jitted graph (contraction regenerates the factors
    from seeds), so a graph that grows with r multiplies XLA compile time: measured
    63 s at r=1 vs 963 s at r=16 per compile on CPU at 12 layers, and 6600 s on an
    A100 at 0.5B. The jaxpr size must not scale with r."""
    from shardes.strategies.lowrank import LowRank

    params = {"w": jnp.zeros((6, 5)), "v": jnp.zeros((4,))}

    def lines(r):
        f = lambda ids: LowRank(r=r).sample(jax.random.key(0), params, ids).factors
        return str(jax.make_jaxpr(f)(jnp.arange(2))).count("\n")

    l1, l8 = lines(1), lines(8)
    assert l8 <= l1 + 8, f"sample's graph grows with rank: {l1} lines at r=1, {l8} at r=8"
