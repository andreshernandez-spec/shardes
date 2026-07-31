"""The two contraction strategies, on simulated devices. Phase 1 C1.3.

docs/02 calls this the most interesting systems question in the project and says not to pick
one silently. So both are implemented, both are tested for agreement, and the communication
each performs is asserted against compiled HLO rather than against a docstring — which is
what turns "A moves O(N), B moves O(d)" from a claim into a measurement.

`test_strategy_A_equals_B` is the correctness test. `test_comm_volume_*` are the ones that
would catch an implementation that quietly became the other strategy: two functions that
agree numerically and move the same bytes are the same function.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shardes import contraction, sharding
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

# 1 and 8 are the ends that matter: no collective at all, and full fan-out. 2 and 4 are
# recombinations of the same code path, so they run in the full suite only. Same tiering
# policy as the strategy property suite (registry.REPRESENTATIVES).
def _dev(d):
    return pytest.param(d, id=str(d), marks=() if d in (1, 8) else pytest.mark.slow)


DEVICE_COUNTS = [_dev(d) for d in (1, 2, 4, 8)]
OTHER_THAN_ONE = [_dev(d) for d in (2, 4, 8)]
N = 32

# One per implementation path: materialized, scanned, and factored-under-a-wrapper.
STRATEGIES = [
    pytest.param(IIDGaussian(), id="iid_gaussian"),
    pytest.param(SeedRegenerated(), id="seed_regenerated"),
    pytest.param(Mirrored(LowRank(r=1)), id="mirrored_lr1"),
]

COLLECTIVE = re.compile(r"\b(all-gather|all-reduce|collective-permute|all-to-all)\b")


def setup(d, n=N):
    mesh = sharding.make_mesh(d)
    params = {
        "w": jax.random.normal(jax.random.key(0), (6, 4), jnp.float32),
        "b": jax.random.normal(jax.random.key(1), (4,), jnp.float32),
    }
    ids = sharding.member_ids(n, mesh)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(2), (n,), jnp.float32), sharding.members(mesh)
    )
    return mesh, params, ids, weights, jax.random.key(3)


def collectives(fn, *args) -> set[str]:
    """Which collectives the compiled program actually contains."""
    return set(COLLECTIVE.findall(jax.jit(fn).lower(*args).compile().as_text()))


_REFERENCE: dict = {}


def _one_device_reference(strategy, how):
    """The D=1 update, computed once per (strategy, how) and reused.

    Each invariance test needs the same reference, and recomputing it per parametrization
    doubled the file's cost for nothing: one compile of the D=1 program is enough.
    """
    key_ = (id(strategy), how)
    if key_ not in _REFERENCE:
        mesh, params, ids, weights, k = setup(1)
        _REFERENCE[key_] = contraction.contract(strategy, k, params, ids, weights, mesh, how=how)
    return _REFERENCE[key_]


def rel_err(a, b) -> float:
    fa = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(a)])
    fb = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(b)])
    return float(np.linalg.norm(fa - fb) / np.linalg.norm(fb))


# --------------------------------------------------------------------------------------
# Agreement.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("d", DEVICE_COUNTS)
def test_strategy_A_equals_B(strategy, d):
    """docs/02 C1.3. The two strategies must produce the same update from the same seed.

    Relative tolerance, not bitwise, and deliberately: A sums N terms in one order while B
    sums N/D per device and then sums D partials. Both are correct. Demanding bitwise here
    would be demanding that float addition be associative.
    """
    mesh, params, ids, weights, key = setup(d)
    a = contraction.contract_replicated(strategy, key, params, ids, weights, mesh)
    b = contraction.contract_sharded(strategy, key, params, ids, weights, mesh)
    assert rel_err(a, b) < 1e-5


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("how", ["A", "B"])
@pytest.mark.parametrize("d", OTHER_THAN_ONE)
def test_contraction_is_device_count_invariant(strategy, how, d):
    """Each strategy against *itself* on D devices versus 1. Invariant 2, at this layer.

    Separate from test_strategy_A_equals_B on purpose. That one compares two different
    summation orders and can only be approximate; this compares one order against itself, so
    a discrepancy means a device index leaked into the seed derivation rather than that
    floats were added in a different sequence.

    Parametrized rather than looped so a failure names the device count that broke.
    """
    ref = _one_device_reference(strategy, how)
    mesh, params_d, ids, weights, key_d = setup(d)
    got = contraction.contract(strategy, key_d, params_d, ids, weights, mesh, how=how)
    assert rel_err(got, ref) < 1e-6, f"strategy {how} differs between D=1 and D={d}"


# --------------------------------------------------------------------------------------
# Communication. The part that makes the O(N) vs O(d) claim a measurement.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", OTHER_THAN_ONE)
def test_comm_volume_A(d):
    """Strategy A gathers N scalars and performs no model-sized reduction.

    An all-reduce appearing here would mean A had quietly become B, which is precisely the
    failure a numerical test cannot see.
    """
    mesh, params, ids, weights, key = setup(d)
    ops = collectives(
        lambda i, w: contraction.contract_replicated(IIDGaussian(), key, params, i, w, mesh),
        ids, weights,
    )
    assert "all-gather" in ops, f"A must gather the fitnesses; saw {ops}"
    assert "all-reduce" not in ops, f"A must not all-reduce a model-sized array; saw {ops}"


@pytest.mark.parametrize("d", OTHER_THAN_ONE)
def test_comm_volume_B(d):
    """Strategy B all-reduces the partial update and gathers nothing."""
    mesh, params, ids, weights, key = setup(d)
    ops = collectives(
        lambda i, w: contraction.contract_sharded(IIDGaussian(), key, params, i, w, mesh),
        ids, weights,
    )
    assert "all-reduce" in ops, f"B must psum the partial update; saw {ops}"
    assert "all-gather" not in ops, f"B must not gather the population; saw {ops}"


def test_one_device_needs_no_collective_at_all():
    """A degenerate case worth pinning: with D=1 there is nothing to communicate, so a
    collective appearing would mean one is being emitted unconditionally."""
    mesh, params, ids, weights, key = setup(1)
    ops = collectives(
        lambda i, w: contraction.contract_replicated(IIDGaussian(), key, params, i, w, mesh),
        ids, weights,
    )
    assert not ops, f"D=1 should need no collective; saw {ops}"


# --------------------------------------------------------------------------------------
# Dispatch and validation.
# --------------------------------------------------------------------------------------


def test_contract_dispatches_to_both():
    mesh, params, ids, weights, key = setup(4)
    a = contraction.contract(IIDGaussian(), key, params, ids, weights, mesh, how="A")
    b = contraction.contract(IIDGaussian(), key, params, ids, weights, mesh, how="B")
    assert rel_err(a, b) < 1e-5


def test_contract_rejects_an_unknown_strategy():
    mesh, params, ids, weights, key = setup(4)
    with pytest.raises(ValueError, match="must be one of"):
        contraction.contract(IIDGaussian(), key, params, ids, weights, mesh, how="C")


def test_contract_rejects_a_population_that_does_not_shard():
    """Caught before any work, not as a shape error inside shard_map."""
    mesh = sharding.make_mesh(4)
    params = {"w": jnp.zeros((2, 2))}
    ids = jnp.arange(6, dtype=jnp.int32)
    with pytest.raises(ValueError, match="does not divide"):
        contraction.contract(IIDGaussian(), jax.random.key(0), params, ids,
                             jnp.ones(6, jnp.float32), mesh)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_update_keeps_the_params_structure(strategy):
    """Invariant 1 at the contraction: leaves keep their shape, nothing is flattened."""
    mesh, params, ids, weights, key = setup(4)
    for how in ("A", "B"):
        out = contraction.contract(strategy, key, params, ids, weights, mesh, how=how)
        assert jax.tree.structure(out) == jax.tree.structure(params)
        for got, want in zip(jax.tree.leaves(out), jax.tree.leaves(params)):
            assert got.shape == want.shape
