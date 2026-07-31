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


# --------------------------------------------------------------------------------------
# Per-coordinate sigma. docs/02 C1.4: the diagonal that shipped instead of sharded state.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_uniform_diagonal_equals_a_scalar_sigma(strategy):
    """The widening must be exactly that: a superset, with the scalar path unchanged.

    A params-shaped sigma whose entries are all `s` has to give bitwise what the scalar `s`
    gives. If it does not, the diagonal is not a generalization of isotropic ES but a
    different algorithm wearing its name.
    """
    mesh = sharding.make_mesh(4)
    p0 = params0()
    scalar = 0.01
    diagonal = jax.tree.map(lambda leaf: jnp.full(leaf.shape, scalar, jnp.float32), p0)

    outs = []
    for sig in (scalar, diagonal):
        es = ShardedES(strategy(), n=N, sigma=sig, lr=0.05, mesh=mesh)
        state = es.init(jax.random.key(0), p0)
        pert, state = es.ask(state)
        fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
        outs.append(es.tell(state, pert, fitness).params)

    assert rel_err(outs[1], outs[0]) == 0.0


def test_a_diagonal_sigma_scales_each_coordinate_separately():
    """The claim the diagonal actually makes: leaf `w` is explored at sigma_w, `b` at sigma_b.

    Asserted at `apply`, which is where sigma enters, and **not** on the size of the step
    `tell` takes. Sigma cancels out of the mean step by construction: the estimator divides
    by `n*sigma`, which is exactly what makes `g_hat` an estimate of `grad f` rather than of
    `sigma * grad f`. A first version of this test asserted that a 100x larger sigma moves a
    leaf further, and it failed in the opposite direction — the small-sigma leaf moved *more*,
    because its update inherits noise from the large-sigma leaf and then divides it by a tiny
    sigma. That is correct behaviour and a good argument for per-coordinate sigmas being about
    conditioning rather than step size.

    Here the model reads one leaf, so the spread of `apply`'s per-member outputs is exactly
    that leaf's sigma times the spread of its perturbation. Doubling the leaf's sigma must
    double the spread, and changing the *other* leaf's sigma must not move it at all.
    """
    mesh = sharding.make_mesh(4)
    p0 = params0()

    def only_w(p, _x):
        return jnp.sum(p["w"])

    def spread(sigma_w, sigma_b):
        sig = {"w": jnp.full(p0["w"].shape, sigma_w, jnp.float32),
               "b": jnp.full(p0["b"].shape, sigma_b, jnp.float32)}
        es = ShardedES(IIDGaussian(), n=512, sigma=sig, lr=0.05, mesh=mesh)
        state = es.init(jax.random.key(0), p0)
        pert, state = es.ask(state)
        return float(jnp.std(es.apply(only_w, state, pert)(jnp.zeros(()))))

    base = spread(0.01, 0.001)
    doubled_w = spread(0.02, 0.001)
    changed_b = spread(0.01, 0.5)

    assert abs(doubled_w / base - 2.0) < 1e-4, (doubled_w, base)
    assert abs(changed_b / base - 1.0) < 1e-4, "sigma_b leaked into leaf w"


def test_the_diagonal_survives_a_sharded_contraction():
    """A per-coordinate sigma must not disturb device invariance.

    `sigma` is replicated state, so it does not interact with the member axis — but it is
    now params-shaped, which is exactly the shape that would be sharded by mistake.
    """
    p0 = params0()
    diagonal = jax.tree.map(lambda leaf: 0.01 * jnp.ones(leaf.shape, jnp.float32), p0)

    def run(d):
        mesh = sharding.make_mesh(d)
        es = ShardedES(IIDGaussian(), n=N, sigma=diagonal, lr=0.05, mesh=mesh)
        state = es.init(jax.random.key(0), p0)
        pert, state = es.ask(state)
        fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
        return es.tell(state, pert, fitness).params

    ref = run(1)
    for d in (2, 8):
        assert rel_err(run(d), ref) < 1e-6, f"diagonal sigma broke invariance at D={d}"


def test_per_leaf_normalizes_scalars_and_passes_trees_through():
    from shardes.strategies._scale import per_leaf

    tree = {"w": jnp.zeros((2, 3)), "b": jnp.zeros((3,))}
    out = per_leaf(0.5, tree)
    assert jax.tree.structure(out) == jax.tree.structure(tree)
    assert all(float(x) == 0.5 for x in jax.tree.leaves(out))

    given = {"w": jnp.ones((2, 3)), "b": 2 * jnp.ones((3,))}
    assert per_leaf(given, tree) is given


# --------------------------------------------------------------------------------------
# Group-relative shaping end to end. Phase 1 C1.6.
# --------------------------------------------------------------------------------------


def multi_task(p, _x):
    """Three tasks on wildly different reward scales, which is the case C1.6 is about.

    Task 0 is the objective; tasks 1 and 2 are the same objective rescaled and offset. Any
    shaping that does not normalize per task will be driven almost entirely by task 2.
    """
    base = sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p))
    return jnp.stack([base, 10.0 * base + 5.0, 5000.0 * base + 12345.0])


@pytest.mark.parametrize("d", OTHER_THAN_ONE)
def test_group_relative_survives_the_sharded_path(d):
    """A 2-D fitness has to reach `tell` intact: the shaping reduces over the *member* axis,
    which is the sharded one, so it is a barrier exactly like centered ranks."""
    from shardes import shaping as shp

    mesh = sharding.make_mesh(d)
    es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh,
                   shaping=shp.group_relative)
    state = es.init(jax.random.key(0), params0())
    pert, state = es.ask(state)
    fitness = es.apply(multi_task, state, pert)(jnp.zeros(()))
    assert fitness.shape == (N, 3)

    out = es.tell(state, pert, fitness)
    assert jax.tree.structure(out.params) == jax.tree.structure(params0())
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(out.params))


def test_group_relative_is_device_count_invariant():
    """It reduces across the sharded axis, so it is the shaping most able to break
    invariance. Checked separately from the scalar-fitness shapings for that reason."""
    from shardes import shaping as shp

    def run(d):
        mesh = sharding.make_mesh(d)
        es = ShardedES(IIDGaussian(), n=N, sigma=0.01, lr=0.05, mesh=mesh,
                       shaping=shp.group_relative)
        state = es.init(jax.random.key(0), params0())
        pert, state = es.ask(state)
        fitness = es.apply(multi_task, state, pert)(jnp.zeros(()))
        return es.tell(state, pert, fitness).params

    ref = run(1)
    for d in (2, 8):
        assert rel_err(run(d), ref) < 1e-6, f"group_relative broke invariance at D={d}"


@pytest.mark.slow
def test_group_relative_is_not_dominated_by_the_largest_scale_task():
    """The end-to-end version of the scale-invariance claim, against a shaping that lacks it.

    `centered` on the summed fitness is driven by task 2, whose rewards are 5000x the others.
    `group_relative` gives the three tasks equal say. Since all three encode the *same*
    objective here, both descend — what differs is that only one of them would still work if
    the tasks disagreed, so this asserts the mechanism rather than the outcome: the
    group-relative weights must be nearly unchanged when task 2 is rescaled again, and the
    summed-fitness weights must not be.
    """
    from shardes import shaping as shp

    mesh = sharding.make_mesh(4)
    es = ShardedES(IIDGaussian(), n=64, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(jax.random.key(0), params0())
    pert, state = es.ask(state)
    f = es.apply(multi_task, state, pert)(jnp.zeros(()))
    rescaled = f * jnp.array([1.0, 1.0, 1e6])

    grouped = (shp.group_relative(f), shp.group_relative(rescaled))
    summed = (shp.centered(jnp.sum(f, axis=1)), shp.centered(jnp.sum(rescaled, axis=1)))

    def drift(pair):
        a, b = pair
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))

    assert drift(grouped) < 1e-3, "group_relative should be immune to a per-task rescale"
    assert drift(summed) > 0.1, "the summed baseline should be sensitive to it"
