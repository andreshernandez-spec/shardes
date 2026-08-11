"""The sharded core, on 8 simulated CPU devices. Phase 1.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/

Sharding logic, PartitionSpec errors, shard_map signatures and collective placement all
reproduce faithfully on CPU. Do not rent a GPU to debug a sharding annotation.

Still to come, from docs/02-phase1-sharded-core.md:

    test_device_invariance          same seed on 1 device and on 8 simulated devices gives
                                    the same update. rtol=1e-12 in f32, near bitwise.
                                    The most important test in the repo.
    test_strategy_A_equals_B        the two contraction strategies produce the same update
                                    for the same seed
    test_comm_volume_A              instrument collectives, assert A moves O(N) not O(Nd)
    test_comm_volume_B              assert B moves exactly one params-sized psum per
                                    generation
    test_state_sharding             distribution state carries the intended NamedSharding,
                                    not replicated

What is here now is C1.2: the mesh, the two shardings, and the seed contract at the layout
level. The property that matters is `test_member_ids_are_global_*` — sharding partitions an
array without renumbering it, so global member indices are a consequence of the data layout
rather than something the code must remember to do.

There were four shardings until `per_member` and `shard_perturbation` were removed, both
dead: one was a spelling of `members`, the other constrained the producer, which does not
distribute anything. `test_the_evaluation_distributes_across_devices` is what covers the
property they appeared to cover and did not.

What simulated devices do not model is interconnect bandwidth or latency. Every correctness
claim is answerable here; no timing claim is.
"""

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from shardes import sharding
from shardes.core import ShardedES
from shardes.problems import transformer_block
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

DEVICE_COUNTS = [1, 2, 4, 8]


@pytest.mark.parametrize("d", DEVICE_COUNTS)
def test_mesh_has_the_requested_device_count(d):
    mesh = sharding.make_mesh(d)
    assert sharding.n_devices(mesh) == d
    assert mesh.axis_names == (sharding.POP,)


def test_mesh_defaults_to_every_visible_device():
    assert sharding.n_devices(sharding.make_mesh()) == len(jax.devices())


def test_mesh_refuses_more_devices_than_exist():
    with pytest.raises(ValueError, match="visible"):
        sharding.make_mesh(len(jax.devices()) + 1)


# --------------------------------------------------------------------------------------
# The seed contract, structurally.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", DEVICE_COUNTS)
def test_member_ids_are_global_regardless_of_device_count(d):
    """The whole trick. `arange(n)` sharded is still `arange(n)`, so device k holds literal
    global indices and a strategy inside shard_map cannot see a local one."""
    mesh = sharding.make_mesh(d)
    ids = sharding.member_ids(64, mesh)
    assert jnp.array_equal(ids, jnp.arange(64, dtype=jnp.int32))
    assert ids.sharding.spec == P(sharding.POP)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_member_ids_land_on_the_right_device(d):
    """Device k must hold [k*n/D, (k+1)*n/D). Asserted per shard rather than on the
    assembled array: the assembled array looks right even if the pieces were permuted, and
    a permutation is exactly what a wrong PartitionSpec produces."""
    mesh = sharding.make_mesh(d)
    n = 64
    ids = sharding.member_ids(n, mesh)
    per = n // d
    for k, shard in enumerate(ids.addressable_shards):
        want = jnp.arange(k * per, (k + 1) * per, dtype=jnp.int32)
        assert jnp.array_equal(shard.data, want), f"device {k} holds {shard.data}, want {want}"


def test_member_ids_are_identical_across_device_counts():
    """Invariant 2 at its cheapest: the ids a strategy sees do not depend on D at all."""
    # numpy, not jnp: two arrays committed to different meshes cannot be compared
    # elementwise under explicit sharding, and the ValueError would read as a test bug
    # rather than as the API refusing a cross-mesh op.
    seen = [np.asarray(sharding.member_ids(128, sharding.make_mesh(d))) for d in DEVICE_COUNTS]
    for other in seen[1:]:
        assert np.array_equal(seen[0], other)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_a_shard_map_sees_global_ids(d):
    """The contract where it is consumed, inside shard_map rather than outside it.

    Each device reports the ids it was handed. If sharding renumbered, or if the code ever
    reached for a device index, the union would not be arange(n).
    """
    mesh = sharding.make_mesh(d)
    n = 32
    ids = sharding.member_ids(n, mesh)
    f = jax.jit(shard_map(lambda x: x, mesh=mesh, in_specs=(P(sharding.POP),),
                          out_specs=P(sharding.POP)))
    assert jnp.array_equal(f(ids), jnp.arange(n, dtype=jnp.int32))


# --------------------------------------------------------------------------------------
# Population validation.
# --------------------------------------------------------------------------------------


def test_uneven_population_is_refused():
    """An uneven split would change the update rather than raise, so it has to raise here."""
    with pytest.raises(ValueError, match="does not divide"):
        sharding.check_population(30, sharding.make_mesh(4))


def test_odd_shard_under_pairing_is_refused():
    """Mirrored pairs (2k, 2k+1). 8 members over 4 devices is 2 each and fine; 4 over 4 is
    1 each and splits every pair."""
    mesh4 = sharding.make_mesh(4)
    sharding.check_population(8, mesh4, paired=True)
    with pytest.raises(ValueError, match="odd"):
        sharding.check_population(4, mesh4, paired=True)


def test_pairing_check_is_opt_in():
    """An unpaired strategy must not pay Mirrored's constraint."""
    sharding.check_population(4, sharding.make_mesh(4))


# --------------------------------------------------------------------------------------
# Shardings.
# --------------------------------------------------------------------------------------


def test_the_two_shardings_have_the_specs_they_claim():
    mesh = sharding.make_mesh(4)
    assert sharding.replicated(mesh).spec == P()
    assert sharding.members(mesh).spec == P(sharding.POP)


@pytest.mark.parametrize("shape", [(16,), (16, 5), (16, 5, 3)])
def test_members_shards_the_leading_axis_at_any_rank(shape):
    """`members` is `P(POP)` with no trailing entries, and that has to hold at every rank.

    `ShardedES.apply` constrains its output with it, and that output is `(n,)` for a scalar
    fitness and `(n, episodes)` for `group_relative`. If the short spec did not pad, the
    multi-episode case would place wrongly rather than fail.

    It is also why `per_member(mesh, rank)` was deleted rather than kept: it returned
    `P(POP, None * rank)`, JAX pads a short spec with None, and the two place identically.
    This test is what makes that a fact rather than a belief.
    """
    mesh = sharding.make_mesh(4)
    x = jax.device_put(jnp.zeros(shape), sharding.members(mesh))
    assert x.sharding.spec == P(sharding.POP)
    # The member axis is split four ways and nothing else is touched.
    assert x.addressable_shards[0].data.shape == (shape[0] // 4, *shape[1:])


# --------------------------------------------------------------------------------------
# G1 exit criterion 6, and invariant 1. Static, so it costs nothing to establish now.
# --------------------------------------------------------------------------------------


def _uses(name: str) -> list[str]:
    """Every real reference to `name` under src/, by parsing rather than grepping.

    Text matching finds these files' own docstrings: core.py says "no ravel_pytree" and
    sharding.py says "NOT jax.experimental.shard_map". A banned-identifier check that trips
    on the sentence explaining the ban is worse than no check, because the obvious fix is to
    weaken the check.
    """
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    hits = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            found = (
                (isinstance(node, ast.Name) and node.id == name)
                or (isinstance(node, ast.Attribute) and node.attr == name)
                or (isinstance(node, (ast.Import, ast.ImportFrom))
                    and any(a.name == name or (a.asname or "") == name for a in node.names))
                or (isinstance(node, ast.ImportFrom) and name in (node.module or ""))
            )
            if found:
                hits.append(f"{path.relative_to(src)}:{node.lineno}")
    return hits


def test_no_ravel_pytree_under_src():
    """Invariant 1 (CLAUDE.md) and Gate G1 criterion 6.

    Nothing under src/ may flatten a solution to one dense vector. That is the architectural
    difference from evosax and the reason low-rank perturbation is expressible at all.
    Static, because by the time a numerical test catches it the API has already been shaped
    around it.
    """
    assert not _uses("ravel_pytree")


def test_the_deprecated_shard_map_path_is_not_used():
    """`jax.experimental.shard_map` is deprecated as of JAX 0.8.0, and EGGROLL's own scripts
    still use it, so it is the natural thing to copy (docs/02, traps)."""
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    bad = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and "experimental" in (node.module or ""):
                if any(a.name == "shard_map" for a in node.names):
                    bad.append(f"{path.relative_to(src)}:{node.lineno}")
    assert not bad, f"deprecated shard_map import: {bad}. Use `from jax import shard_map`."


# --------------------------------------------------------------------------------------
# The evaluation has to actually distribute. Invariant 2 says the update is the same on
# 1 and 8 devices; it says nothing about the work being divided, and for a long time it
# was not.
# --------------------------------------------------------------------------------------


def _one_score_per_episode(params, x):
    """A model scoring several episodes per member, so the fitness is `(n, episodes)`.

    `group_relative` is the shaping that consumes this shape. Its correctness through the
    sharded path is covered in `tests/test_core.py`; what is covered here is that the shape
    still *distributes*, which no correctness test can see.
    """
    return jnp.stack([transformer_block.loss(params, x)] * 3)


@pytest.mark.parametrize("d", [2, 4, 8])
@pytest.mark.parametrize(
    "model, fitness_shape",
    [(transformer_block.loss, "(n,)"),
     (_one_score_per_episode, "(n, episodes)")],
)
def test_the_evaluation_distributes_across_devices(d, model, fitness_shape):
    """Per-device FLOPs of the compiled evaluation fall as 1/D, or nothing was sharded.

    **This is the test the project did not have, and its absence cost a rented 8-GPU
    session.** The mesh is `AxisType.Auto`, so sharding is propagated rather than declared,
    and for a long time the only constraint on the fitness was `tell`'s `replicated`, needed
    for the global sort in `centered_ranks`. That propagates backwards: the cheapest way to
    have the whole fitness on every device is to compute it on every device, needing no
    collective at all. So every device evaluated the whole population, per-device FLOPs did
    not move with `D`, and parallel efficiency was exactly `1/D`. The 2026-08-06 sweep
    measured 0.112 to 0.142 at `D=8` against `1/8 = 0.125` and nothing failed, because every
    correctness test still passed: the update was right, it was just computed eight times.

    `docs/diagnosis-replicated-evaluation.md` has the full account. `ShardedES.apply` now
    constrains its output to the member axis, which is what forces the vmap to partition.

    **Both fitness shapes, because the constraint is `P("pop")` with no trailing entries.**
    A scalar fitness is `(n,)` and `group_relative`'s is `(n, episodes)`, and the short spec
    covers the second only because JAX pads it with None. A regression that replicated the
    multi-episode path would otherwise pass every test in the suite: `test_core.py` checks
    that `group_relative` survives sharding and stays device-count invariant, and both of
    those are true of a computation that runs whole on every device.

    Asserted on the compiled program rather than on wall clock, so it holds on simulated
    devices and cannot go quiet on a machine with one GPU.
    """
    mesh = sharding.make_mesh(d)
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=16)
    data = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=16,
                                        batch=2, seq=4)
    n = 8 * d

    def eval_flops(devices):
        m = sharding.make_mesh(devices)
        es = ShardedES(IIDGaussian(), n=n, sigma=0.01, lr=0.05, mesh=m, how="A")
        state = es.init(key, params)

        def evaluate(s):
            pert, scaled = es.ask(s)
            return es.apply(model, scaled, pert)(data)

        analysis = jax.jit(evaluate).lower(state).compile().cost_analysis()
        analysis = analysis[0] if isinstance(analysis, list) else analysis
        return float(analysis["flops"])

    one, many = eval_flops(1), eval_flops(d)
    ratio = many / one
    assert ratio < 1.5 / d, (
        f"per-device evaluation FLOPs went {one:,.0f} -> {many:,.0f} from D=1 to D={d} with "
        f"a {fitness_shape} fitness, a ratio of {ratio:.3f} against an ideal of "
        f"{1 / d:.3f}. The population is not being divided: every device is evaluating all "
        "of it, so wall clock cannot fall with D."
    )


#: Every strategy in the library, including the two wrappers. The point of the list is that
#: `test_every_strategy_evaluates_only_its_own_shard` runs over all of it: the defect it
#: guards is strategy-dependent, and the test that came before it hardcoded one strategy.
ALL_STRATEGIES = [
    ("iid_gaussian", IIDGaussian),
    ("seed_regenerated", SeedRegenerated),
    ("lowrank_r1", lambda: LowRank(r=1)),
    ("mirrored_lowrank", lambda: Mirrored(LowRank(r=1))),
    ("mirrored_seed_regenerated", lambda: Mirrored(SeedRegenerated())),
]


@pytest.mark.parametrize("d", [2, 4, 8])
@pytest.mark.parametrize("name, make", ALL_STRATEGIES, ids=[n for n, _ in ALL_STRATEGIES])
def test_every_strategy_evaluates_only_its_own_shard(d, name, make):
    """Each device hands its strategy `n/D` member ids, whatever the strategy does with them.

    **This is the second time the evaluation silently failed to distribute, and the first
    test could not see it.** `test_the_evaluation_distributes_across_devices` above asserts
    that per-device FLOPs fall as `1/D`, which caught the original bug and would catch a
    regression in any strategy that evaluates with a `vmap`. It is **blind to `lax.scan`**:
    XLA lowers a scan to a `while` loop and `cost_analysis` counts the loop body once,
    ignoring the trip count. Measured, at `d_model=128, n=64`: a device scanning 8 members
    and a device scanning 64 report 10,308,543 and 10,308,533 FLOPs. The ratio is 1.000
    whether the scan was partitioned or not, so that assertion cannot distinguish the two,
    in either direction.

    `SeedRegenerated` evaluates with a scan, and it went on evaluating the whole population
    on every device through the fix that repaired the other three strategies. Parallel
    efficiency stayed at `1/D` and wall clock stayed flat at 431.95, 435.29, 436.43,
    440.35 ms across `D=1,2,4,8` on 8x A100, which is how it was eventually found: by a
    rented sweep, again (`docs/diagnosis-seed-regenerated-scan.md`).

    So this asserts the property directly rather than through a proxy the compiler is free
    to be vague about. `ShardedES.apply` runs the evaluation under `shard_map`, so the
    `member_ids` its strategy receives is that device's shard and nothing else. Length
    `n/D` is what "the population is divided" *means*; every downstream claim, FLOPs
    included, is a consequence.

    It is a white-box test: it reaches into what `apply` passes the strategy. That is
    deliberate. The black-box measures available on simulated devices are wall clock, which
    is meaningless when eight devices share four cores, and FLOPs, which is the metric that
    just failed to see a whole class of bug.
    """
    mesh = sharding.make_mesh(d)
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=16)
    data = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=16,
                                        batch=2, seq=4)
    n = 8 * d

    strategy = make()
    seen: list[int] = []
    inner = type(strategy).apply

    def record(self, model, params_, pert, sigma):
        seen.append(pert.member_ids.shape[0])
        return inner(self, model, params_, pert, sigma)

    es = ShardedES(strategy, n=n, sigma=0.01, lr=0.05, mesh=mesh, how="A")
    state = es.init(key, params)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(type(strategy), "apply", record)
    try:
        def evaluate(s):
            pert, scaled = es.ask(s)
            return es.apply(transformer_block.loss, scaled, pert)(data)

        jax.jit(evaluate).lower(state)
    finally:
        monkey.undo()

    assert seen, "the strategy's apply was never called; this test no longer tests anything"
    assert set(seen) == {n // d}, (
        f"{name} was handed member_ids of length {sorted(set(seen))} at D={d}, expected "
        f"{n // d} = n/D. A strategy that receives all {n} ids evaluates the whole "
        "population on every device, so wall clock cannot fall with D no matter what the "
        "FLOP counters say."
    )
