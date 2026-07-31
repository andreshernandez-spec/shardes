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

What is here now is C1.2: the mesh, the four shardings, and the seed contract at the layout
level. The property that matters is `test_member_ids_are_global_*` — sharding partitions an
array without renumbering it, so global member indices are a consequence of the data layout
rather than something the code must remember to do.

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


def test_the_four_shardings_have_the_specs_they_claim():
    mesh = sharding.make_mesh(4)
    assert sharding.replicated(mesh).spec == P()
    assert sharding.members(mesh).spec == P(sharding.POP)
    assert sharding.per_member(mesh, 0).spec == P(sharding.POP)
    assert sharding.per_member(mesh, 2).spec == P(sharding.POP, None, None)


@pytest.mark.parametrize("d", DEVICE_COUNTS)
def test_shard_perturbation_splits_member_axes_and_replicates_the_rest(d):
    """A materialized perturbation shards on its leading axis; a `like` reference does not.

    SeedRegenerated carries unbatched params alongside batched state, so both cases live in
    one tree and placement has to be decided per leaf.
    """
    mesh = sharding.make_mesh(d)
    n = 16
    tree = {
        "eps": jnp.zeros((n, 5, 3)),     # per member
        "like": jnp.zeros((5, 3)),       # replicated reference, no member axis
        "scalar": jnp.float32(1.0),
    }
    out = sharding.shard_perturbation(tree, mesh, n)
    assert out["eps"].sharding.spec == P(sharding.POP, None, None)
    assert out["like"].sharding.spec == P()
    assert out["scalar"].sharding.spec == P()


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
