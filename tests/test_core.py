"""ask / eval / tell end to end. Phase 1 C1.1.

The headline is `test_device_invariance`: the same seed on 1 device and on 8 simulated
devices must produce the same update. CLAUDE.md calls it the most important test in the
repo, and it is the one every Phase 0 design decision was made to permit — the seed contract
exists so that it can pass, and `member_ids` is `arange(n)` sharded so that it passes by
construction rather than by care.

It is a *near-bitwise* claim, unlike `test_strategy_A_equals_B`, because it compares one
summation order against itself. A real failure means a device index leaked into seed
derivation, not that floats were added in a different sequence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shardes import sharding
from shardes.core import ShardedES, State
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

N = 32


def _dev(d):
    """1 and 8 are the ends that matter; 2 and 4 exercise the same path."""
    return pytest.param(d, id=str(d), marks=() if d in (1, 8) else pytest.mark.slow)


DEVICE_COUNTS = [_dev(d) for d in (1, 2, 4, 8)]
OTHER_THAN_ONE = [_dev(d) for d in (2, 4, 8)]

STRATEGIES = [
    pytest.param(IIDGaussian, id="iid_gaussian"),
    pytest.param(SeedRegenerated, id="seed_regenerated"),
    pytest.param(lambda: Mirrored(LowRank(r=1)), id="mirrored_lr1"),
]


def params0():
    k1, k2 = jax.random.split(jax.random.key(0))
    return {
        "w": jax.random.normal(k1, (6, 4), jnp.float32),
        "b": jax.random.normal(k2, (4,), jnp.float32),
    }


def sphere(p, _x):
    """Sum of squares. Minimized at zero, so descent must shrink it."""
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p))


def flat(tree) -> np.ndarray:
    """Host-side flatten, for comparison only. Never used inside src/ (invariant 1)."""
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


def rel_err(a, b) -> float:
    fa, fb = flat(a), flat(b)
    return float(np.linalg.norm(fa - fb) / np.linalg.norm(fb))


_REFERENCE: dict = {}


def one_device_reference(strategy_factory, how, ident):
    """The D=1 update, computed once per (strategy, how).

    Every device count compares against the same reference, so recomputing it per
    parametrization pays for the same compile three times over.
    """
    if (ident, how) not in _REFERENCE:
        _REFERENCE[(ident, how)] = run_one_generation(strategy_factory, 1, how=how)
    return _REFERENCE[(ident, how)]


def run_one_generation(strategy_factory, d, *, how="B", n=N, seed=0):
    """One full ask/eval/tell on `d` devices. Returns the updated params."""
    mesh = sharding.make_mesh(d)
    es = ShardedES(strategy_factory(), n=n, sigma=0.01, lr=0.05, mesh=mesh, how=how)
    state = es.init(jax.random.key(seed), params0())
    pert, state = es.ask(state)
    fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
    return es.tell(state, pert, fitness).params


# --------------------------------------------------------------------------------------
# The one that matters.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("how", ["A", "B"])
@pytest.mark.parametrize("d", OTHER_THAN_ONE)
def test_device_invariance(strategy, how, d):
    """Invariant 2, and Gate G1 criterion 2. Same seed, D devices vs 1, same update.

    Tolerance is 1e-6 rather than the 1e-12 docs/02 quotes, and the difference is the
    contraction, not the seeds: Strategy B sums n/D partials per device and then D of them,
    so the arithmetic genuinely reassociates as D changes. A seed leak would show up orders
    of magnitude larger than this — the failure mode is a *different population*, not a
    different rounding.
    """
    ref = one_device_reference(strategy, how, strategy)
    got = run_one_generation(strategy, d, how=how)
    assert rel_err(got, ref) < 1e-6, f"{how} on {d} devices differs from 1 device"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_population_itself_is_device_count_invariant(strategy):
    """One level below the update: the perturbation `ask` produces must not depend on D.

    Separate from test_device_invariance so a failure says *which* half broke. If this
    passes and that fails, the contraction is at fault; if this fails, the seed contract is.
    """
    def epsilon(d):
        mesh = sharding.make_mesh(d)
        es = ShardedES(strategy(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
        state = es.init(jax.random.key(0), params0())
        pert, _ = es.ask(state)
        onehot = (pert.member_ids == 7).astype(jnp.float32)
        return es.strategy.contract(pert, onehot)

    ref = epsilon(1)
    for d in (2, 4, 8):
        assert rel_err(epsilon(d), ref) < 1e-6, f"member 7 differs at D={d}"


# --------------------------------------------------------------------------------------
# ask / tell behaviour.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ask_returns_a_perturbation_not_parameter_trees(strategy):
    """C1.1, the API decision the whole library hangs on.

    The claim is about the *type*, not about memory. `ask` returns something carrying
    regeneration state, whose structure is not the params tree; a batch of parameter trees
    would have the params structure with a leading member axis and nothing else, and there
    would be nowhere to put `base_key`, so Strategy A and SeedRegenerated would both be
    inexpressible.

    It is deliberately not "no (n, *params) array appears". IIDGaussian materializes exactly
    that, on purpose — it is the memory-hungry reference the other strategies are checked
    against. Invariant 3 bans materialization *under the low-rank path*, and
    test_lowrank.py::test_never_materializes_the_perturbation is where that is asserted, by
    tracing the jaxpr rather than by inspecting shapes.
    """
    mesh = sharding.make_mesh(4)
    es = ShardedES(strategy(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(jax.random.key(0), params0())
    pert, _ = es.ask(state)

    assert hasattr(pert, "base_key") and hasattr(pert, "member_ids")
    assert jnp.shape(pert.member_ids) == (N,)
    assert jax.tree.structure(pert) != jax.tree.structure(params0()), (
        "ask returned something with the params tree structure; it must return a "
        "Perturbation carrying regeneration state"
    )


def test_the_low_rank_path_never_materializes_under_ask():
    """Invariant 3, at the core rather than at the strategy.

    test_lowrank.py checks this inside `apply`'s jaxpr. Here it is the weaker but distinct
    claim that `ask` itself does not hand back a materialized per-member weight: the factors
    are (n, m, r) and (n, k, r), and the (n, m, k) product is what must never appear.
    """
    mesh = sharding.make_mesh(4)
    es = ShardedES(Mirrored(LowRank(r=1)), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(jax.random.key(0), params0())
    pert, _ = es.ask(state)

    shapes = [jnp.shape(x) for x in jax.tree.leaves(pert)]
    assert (N, 6, 4) not in shapes and (N // 2, 6, 4) not in shapes, shapes


def test_ask_advances_the_generation_and_the_population_changes():
    """Two generations must not sample the same members. The counter is what prevents it,
    and a caller cannot forget to advance it because there is no key argument to reuse."""
    mesh = sharding.make_mesh(4)
    es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    s0 = es.init(jax.random.key(0), params0())
    p1, s1 = es.ask(s0)
    p2, s2 = es.ask(s1)

    assert int(s0.generation) == 0 and int(s1.generation) == 1 and int(s2.generation) == 2
    w = jax.random.normal(jax.random.key(1), (N,), jnp.float32)
    assert rel_err(es.strategy.contract(p1, w), es.strategy.contract(p2, w)) > 0.1


def test_ask_is_pure():
    """Same state in, same perturbation out, and the input state is unchanged."""
    mesh = sharding.make_mesh(4)
    es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    s0 = es.init(jax.random.key(0), params0())
    w = jax.random.normal(jax.random.key(1), (N,), jnp.float32)

    a, _ = es.ask(s0)
    b, _ = es.ask(s0)
    assert rel_err(es.strategy.contract(a, w), es.strategy.contract(b, w)) == 0.0
    assert int(s0.generation) == 0


@pytest.mark.slow
def test_tell_descends_on_the_objective():
    """The sign convention, pinned. `tell` minimizes what the fitness measures.

    Asserted rather than documented alone: a flipped sign here is the single easiest bug to
    ship, it produces a run that trains smoothly in the wrong direction, and no shape or
    sharding check would catch it.
    """
    mesh = sharding.make_mesh(4)
    es = ShardedES(Mirrored(IIDGaussian()), n=256, sigma=0.01, lr=0.5, mesh=mesh,
                   shaping=lambda f: f - jnp.mean(f))
    state = es.init(jax.random.key(0), params0())
    before = float(sphere(state.params, None))
    for _ in range(5):
        pert, state = es.ask(state)
        fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
        state = es.tell(state, pert, fitness)
    assert float(sphere(state.params, None)) < before


def test_tell_preserves_the_params_structure():
    """Invariant 1 end to end: what goes in as a pytree comes out as the same pytree."""
    mesh = sharding.make_mesh(4)
    es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(jax.random.key(0), params0())
    pert, state = es.ask(state)
    fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
    out = es.tell(state, pert, fitness)

    assert jax.tree.structure(out.params) == jax.tree.structure(params0())
    for got, want in zip(jax.tree.leaves(out.params), jax.tree.leaves(params0())):
        assert got.shape == want.shape


# --------------------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------------------


def test_the_two_published_algorithms_are_one_argument_apart():
    """C1.5. If switching between Qiu and EGGROLL is not close to this, the abstraction
    failed and should be reworked before going further."""
    mesh = sharding.make_mesh(4)
    common = dict(n=16, sigma=0.01, lr=0.05, mesh=mesh)

    qiu = ShardedES(strategy=Mirrored(SeedRegenerated()), **common)
    eggroll = ShardedES(strategy=Mirrored(LowRank(r=1)), **common)

    for es in (qiu, eggroll):
        state = es.init(jax.random.key(0), params0())
        pert, state = es.ask(state)
        fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
        out = es.tell(state, pert, fitness)
        assert jax.tree.structure(out.params) == jax.tree.structure(params0())


def test_mirrored_population_must_pair_within_a_device():
    """Caught at construction, where the message can name the configuration.

    4 members over 4 devices is one each, so every antithetic pair straddles a boundary.
    That does not raise anywhere downstream; it just quietly stops cancelling.
    """
    mesh = sharding.make_mesh(4)
    with pytest.raises(ValueError, match="odd"):
        ShardedES(Mirrored(IIDGaussian()), n=4, sigma=0.01, lr=0.05, mesh=mesh)
    ShardedES(Mirrored(IIDGaussian()), n=8, sigma=0.01, lr=0.05, mesh=mesh)


def test_population_must_divide_across_the_mesh():
    mesh = sharding.make_mesh(4)
    with pytest.raises(ValueError, match="does not divide"):
        ShardedES(IIDGaussian(), n=30, sigma=0.01, lr=0.05, mesh=mesh)


def test_unknown_contraction_strategy_is_refused():
    mesh = sharding.make_mesh(4)
    with pytest.raises(ValueError, match="how must be one of"):
        ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh, how="C")


def test_state_is_a_pytree_and_survives_a_roundtrip():
    """State crosses jit boundaries, so it has to be a pytree with no hidden Python."""
    mesh = sharding.make_mesh(4)
    es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(jax.random.key(0), params0())
    leaves, treedef = jax.tree.flatten(state)
    assert isinstance(jax.tree.unflatten(treedef, leaves), State)
