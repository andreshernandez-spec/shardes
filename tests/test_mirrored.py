"""Mirrored: antithetic pairs.

The property suite already covers the seed contract, chunk additivity and linearity for
every registered strategy, including the mirrored ones. What is here is specific to the
antithetic structure: that pairs really are negatives, that the variance reduction is
real, and that an odd member count is refused rather than silently mispaired.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import metrics
from shardes import shaping as shp
from shardes.estimator import estimate
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored

RTOL = 1e-6


@pytest.fixture
def params():
    k1, k2 = jax.random.split(jax.random.key(0))
    return {
        "w": jax.random.normal(k1, (6, 4), dtype=jnp.float32),
        "b": jax.random.normal(k2, (4,), dtype=jnp.float32),
    }


def epsilon(strategy, key, params, i, ids):
    pert = strategy.sample(key, params, ids)
    return strategy.contract(pert, (ids == i).astype(jnp.float32))


@pytest.mark.parametrize("inner", [IIDGaussian(), LowRank(r=1)], ids=["full", "lr1"])
def test_pairs_are_exact_negatives(params, inner):
    """Member 2k+1 is -eps_k, not merely another draw. Exact, not approximate."""
    s, key, ids = Mirrored(inner), jax.random.key(1), jnp.arange(8)
    for k in range(4):
        plus = epsilon(s, key, params, 2 * k, ids)
        minus = epsilon(s, key, params, 2 * k + 1, ids)
        negated = jax.tree.map(lambda z: -z, minus)
        assert float(metrics.relative_mse(plus, negated)) == 0.0, f"pair {k}"


def test_uniform_weights_cancel_exactly(params):
    """Constant fitness means no update. This is correct rather than a degenerate case
    to work around, and it is why the unit-scale property test uses random weights."""
    s, ids = Mirrored(IIDGaussian()), jnp.arange(16)
    out = s.contract(s.sample(jax.random.key(2), params, ids), jnp.ones(16))
    for leaf in jax.tree.leaves(out):
        assert float(jnp.max(jnp.abs(leaf))) == 0.0


@pytest.mark.parametrize("n", [1, 3, 5, 15])
def test_odd_member_count_is_refused(params, n):
    """An odd count, or an odd chunk, splits a pair and loses the cancellation. It has to
    fail loudly: a mispaired batch still produces plausible numbers."""
    s = Mirrored(IIDGaussian())
    with pytest.raises(ValueError, match="even member count"):
        s.sample(jax.random.key(3), params, jnp.arange(n))


def test_half_as_many_distinct_directions(params):
    """n members, n/2 draws. That is the cost side of the trade: the effective population
    is halved, which is why comparing against unmirrored i.i.d. at equal n is the honest
    baseline rather than a favourable one."""
    s, ids = Mirrored(IIDGaussian()), jnp.arange(10)
    pert = s.sample(jax.random.key(4), params, ids)
    assert pert.inner.eps["w"].shape[0] == 5
    assert pert.member_ids.shape[0] == 10


def test_cancels_the_constant_term(params):
    """docs/01: on an odd f, mirrored variance is far below i.i.d.

    The mechanism is that f(theta+sigma e) - f(theta-sigma e) removes every even-order
    term, and for raw fitness the dominant noise *is* the even-order constant f(theta).
    Measured below as a variance ratio.
    """
    n, sigma, reps = 32, 0.05, 256
    g = jax.random.normal(jax.random.key(5), (4,), dtype=jnp.float32)
    p = {"theta": jax.random.normal(jax.random.key(6), (4,), dtype=jnp.float32)}
    # Linear plus a large constant: the constant is pure variance for the plain estimator
    # and exactly cancelled by the antithetic pair.
    model = lambda pp, _b: jnp.dot(g, pp["theta"]) + 100.0
    truth = {"theta": g}
    ids = jnp.arange(n)

    def spread(strategy):
        gs = jax.vmap(
            lambda k: estimate(strategy, model, p, None, k, member_ids=ids,
                               sigma=sigma, shaping=shp.none)
        )(jax.random.split(jax.random.key(7), reps))
        return float(jnp.mean(jnp.var(gs["theta"], axis=0)))

    plain = spread(IIDGaussian())
    mirrored = spread(Mirrored(IIDGaussian()))
    assert mirrored < plain / 100, f"mirrored={mirrored:.4g} plain={plain:.4g}"

    # And it is still pointing the right way, not merely quiet.
    gs = jax.vmap(
        lambda k: estimate(Mirrored(IIDGaussian()), model, p, None, k, member_ids=ids,
                           sigma=sigma, shaping=shp.none)
    )(jax.random.split(jax.random.key(8), reps))
    mean = jax.tree.map(lambda z: jnp.mean(z, 0), gs)
    assert float(metrics.cosine_similarity(mean, truth)) > 0.99
