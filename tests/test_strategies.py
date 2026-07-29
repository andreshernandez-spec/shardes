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

import jax
import jax.numpy as jnp
import pytest

from shardes.strategies.protocol import Perturbation, PerturbationStrategy
from shardes.strategies.registry import STRATEGIES

RTOL = 1e-6


def rel_err(got, want):
    flat_g = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(got)])
    flat_w = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(want)])
    return jnp.linalg.norm(flat_g - flat_w) / jnp.linalg.norm(flat_w)


def epsilon(strategy, base_key, params, member_id, member_ids):
    """Member `member_id`'s perturbation, read out via a one-hot contraction.

    `member_ids` is the batch it was sampled as part of, which is exactly the thing the
    seed contract says must not matter.
    """
    pert = strategy.sample(base_key, params, member_ids)
    onehot = (member_ids == member_id).astype(jnp.float32)
    return strategy.contract(pert, onehot)


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
    """Three steps, no more. A fourth would mean the abstraction leaked."""
    methods = {n for n in vars(PerturbationStrategy) if not n.startswith("_")}
    assert methods == {"sample", "apply", "contract"}


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
    [pytest.param(build(), id=name) for name, build in STRATEGIES.items()]
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
    alone = epsilon(strategy, key, params, 7, jnp.array([7]))
    in_batch = epsilon(strategy, key, params, 7, jnp.arange(100))
    shuffled = epsilon(strategy, key, params, 7, jnp.array([13, 7, 2, 99]))
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

    whole = strategy.contract(strategy.sample(key, params, ids), w)
    lo = strategy.contract(strategy.sample(key, params, ids[:6]), w[:6])
    hi = strategy.contract(strategy.sample(key, params, ids[6:]), w[6:])
    assert rel_err(jax.tree.map(jnp.add, lo, hi), whole) < RTOL


@parametrize
def test_contract_is_linear_in_weights(strategy, params):
    key, ids = jax.random.key(4), jnp.arange(12)
    pert = strategy.sample(key, params, ids)
    w1 = jax.random.normal(jax.random.key(5), (12,), dtype=jnp.float32)
    w2 = jax.random.normal(jax.random.key(6), (12,), dtype=jnp.float32)
    a, b = 2.0, -0.5

    got = strategy.contract(pert, a * w1 + b * w2)
    want = jax.tree.map(
        lambda x, y: a * x + b * y,
        strategy.contract(pert, w1),
        strategy.contract(pert, w2),
    )
    assert rel_err(got, want) < RTOL


@parametrize
def test_contract_preserves_pytree_structure(strategy, params):
    """Update tree matches params tree, leaf for leaf. No global flattening."""
    key, ids = jax.random.key(7), jnp.arange(4)
    out = strategy.contract(strategy.sample(key, params, ids), jnp.ones(4))
    assert jax.tree.structure(out) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(params)):
        assert a.shape == b.shape


@parametrize
def test_sample_is_deterministic(strategy, params):
    """Same key, same ids, same perturbation. An experiment you cannot re-run is not one."""
    key, ids = jax.random.key(8), jnp.arange(10)
    a = strategy.contract(strategy.sample(key, params, ids), jnp.ones(10))
    b = strategy.contract(strategy.sample(key, params, ids), jnp.ones(10))
    assert rel_err(a, b) == 0.0


@parametrize
def test_perturbation_is_unit_scale(strategy, params):
    """Per-leaf second moment near 1. Sigma is applied by the core, never here.

    Loose tolerance on purpose: this catches a missing or doubled scale factor, not a
    subtly wrong distribution. Distributional correctness is test_estimator's job.
    """
    key = jax.random.key(9)
    ids = jnp.arange(2048)
    pert = strategy.sample(key, params, ids)
    # (1/sqrt(n)) * sum_i eps_i has unit second moment per element when the eps_i do.
    # Aggregate rather than per-member, so this stays cheap at n = 2048.
    scaled = strategy.contract(pert, jnp.ones(len(ids)) / jnp.sqrt(len(ids)))
    for leaf in jax.tree.leaves(scaled):
        assert 0.5 < float(jnp.sqrt(jnp.mean(leaf**2))) < 2.0


@parametrize
def test_strategy_conforms_to_the_protocol(strategy):
    """Structural check. runtime_checkable only verifies the methods exist, which is
    exactly the failure the signature guards above cannot see: a strategy that forgot
    `contract` entirely."""
    assert isinstance(strategy, PerturbationStrategy)


@parametrize
def test_perturbation_conforms_to_the_protocol(strategy, params):
    pert = strategy.sample(jax.random.key(10), params, jnp.arange(3))
    assert hasattr(pert, "base_key") and hasattr(pert, "member_ids")
    assert jnp.array_equal(pert.member_ids, jnp.arange(3))
