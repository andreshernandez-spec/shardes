"""The ES gradient estimator, against the quadratic's analytic gradient.

The quadratic is the unbiasedness oracle because the estimator is exact there at *any*
sigma: `E[f(theta + sigma eps) eps] = sigma H theta`, since the third moment of a symmetric
distribution vanishes. So a bias that does not go to zero is a bug, not a small-sigma
artifact.

Parameters are tuned, not guessed. At d=8, n=128, sigma=1.0, R=10000 the measured bias is
0.0053, a 4x margin under the 2% gate in docs/01 C0.5. Raw fitness is intrinsically noisy
here because the constant term f(theta) multiplies eps and averages to zero only slowly;
`centered` is roughly 2x tighter for the same cost.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import metrics
from shardes import shaping as shp
from shardes.estimator import estimate
from shardes.problems import quadratic
from shardes.strategies.registry import STRATEGIES

D, N, SIGMA, R = 8, 128, 1.0, 10_000
BIAS_GATE = 0.02  # docs/01 C0.5

strategies = pytest.mark.parametrize(
    "strategy",
    [pytest.param(e.build(), id=name) for name, e in STRATEGIES.items()]
    or [pytest.param(None, id="none", marks=pytest.mark.skip(reason="no strategy registered"))],
)


@pytest.fixture(scope="module")
def problem():
    q = quadratic.make(jax.random.key(0), D, condition_number=10.0)
    theta = jax.random.normal(jax.random.key(1), (D,), dtype=jnp.float32)
    return q, theta, quadratic.grad(q, theta), (lambda p, _x: quadratic.value(q, p))


def run(strategy, problem, shaping, *, n=N, sigma=SIGMA, replicates=R, chunk=None, seed=2):
    _q, theta, _truth, model = problem
    ids = jnp.arange(n)

    def one(key):
        return estimate(strategy, model, theta, jnp.float32(0.0), key,
                        member_ids=ids, sigma=sigma, shaping=shaping, chunk=chunk)

    return jax.vmap(one)(jax.random.split(jax.random.key(seed), replicates))


@pytest.mark.slow
@strategies
@pytest.mark.parametrize("shaping", [shp.none, shp.centered], ids=["none", "centered"])
def test_unbiased_on_the_quadratic(strategy, problem, shaping):
    """mean(g_hat) -> H theta. The measurement Gate G0 rests on."""
    _q, _theta, truth, _model = problem
    got = jnp.mean(run(strategy, problem, shaping), axis=0)
    assert float(metrics.relative_bias(got, truth)) < BIAS_GATE


@pytest.mark.slow
@strategies
@pytest.mark.parametrize("sigma", [0.25, 1.0, 4.0])
def test_unbiased_at_every_sigma(strategy, problem, sigma):
    """Exact at any sigma for a quadratic, so sigma must not shift the mean.

    On a general objective the estimator targets the *smoothed* gradient and sigma does
    matter; that is a property of the objective, not of the estimator.
    """
    _q, _theta, truth, _model = problem
    got = jnp.mean(run(strategy, problem, shp.centered, sigma=sigma), axis=0)
    assert float(metrics.relative_bias(got, truth)) < BIAS_GATE


@strategies
def test_no_sign_flip(strategy, problem):
    """`model` returns the objective and g_hat estimates grad f of it. No maximization
    convention baked in, so the cosine against the analytic gradient is positive."""
    _q, _theta, truth, _model = problem
    got = jnp.mean(run(strategy, problem, shp.centered, replicates=200), axis=0)
    assert float(metrics.cosine_similarity(got, truth)) > 0.9


@strategies
def test_chunked_matches_unchunked(strategy, problem):
    """Contraction Strategy A against Strategy B, on one device.

    chunk=None keeps the perturbation and contracts once. An int re-derives each chunk
    from (key, member_ids) in a second pass. They must agree, which is Phase 1's
    test_strategy_A_equals_strategy_B available now.
    """
    n, chunks = 16, (1, 3, 8, 16)

    # chunk=3 leaves a ragged final chunk, which is the only place the padding can hide a
    # wrong answer. Asserted rather than assumed, so nobody tidies the list into divisors
    # and silently drops the coverage. Padded slots carry weight zero, and a multiply by
    # 0.0 is exact, so their contribution is zero rather than merely small.
    assert any(n % c for c in chunks), "no ragged chunk: the padding path is untested"

    whole = run(strategy, problem, shp.centered_ranks, n=n, replicates=32, chunk=None)
    for chunk in chunks:
        part = run(strategy, problem, shp.centered_ranks, n=n, replicates=32, chunk=chunk)
        assert float(metrics.relative_mse(part, whole)) < 1e-8, f"chunk={chunk}"


@strategies
def test_chunking_does_not_change_the_shaping(strategy, problem):
    """The reason chunking needs two passes: centered ranks are global.

    A chunked run that shaped each chunk independently would rank within chunks, which is
    a different and wrong update. With chunk=1 every weight would be identical.
    """
    n = 16
    whole = run(strategy, problem, shp.centered_ranks, n=n, replicates=8, chunk=None)
    per_one = run(strategy, problem, shp.centered_ranks, n=n, replicates=8, chunk=1)
    assert float(metrics.relative_mse(per_one, whole)) < 1e-8


@pytest.mark.slow
@strategies
def test_centered_ranks_is_not_an_unbiased_estimator(strategy, problem):
    """Asserted, not tolerated.

    Rank shaping is a deliberately different update direction, not an estimator of grad f,
    so its bias does not go to zero. docs/01 C0.5's bias check is a correctness gate only
    on the shaping=none slice; elsewhere it is descriptive. It should still point the
    right way.
    """
    _q, _theta, truth, _model = problem
    got = jnp.mean(run(strategy, problem, shp.centered_ranks), axis=0)
    assert float(metrics.relative_bias(got, truth)) > 0.5
    assert float(metrics.cosine_similarity(got, truth)) > 0.9


@pytest.mark.slow
@strategies
def test_naive_mean_subtraction_would_be_biased(strategy, problem):
    """Pins why `centered` carries the n/(n-1) factor.

    Without it the estimator targets (1 - 1/n) grad f, because f_bar contains f_i and
    correlates with eps_i. At n = 30 that is a 3.3% systematic underestimate that reads
    as a slightly wrong learning rate.
    """
    _q, _theta, truth, _model = problem
    n = 16
    naive = lambda f: f - jnp.mean(f)
    got = jnp.mean(run(strategy, problem, naive, n=n, replicates=40_000), axis=0)

    ratio = float(jnp.median(got / truth))
    assert abs(ratio - (1 - 1 / n)) < 0.02, f"expected ~{1 - 1 / n}, got {ratio}"

    corrected = jnp.mean(run(strategy, problem, shp.centered, n=n, replicates=40_000), axis=0)
    assert float(metrics.relative_bias(corrected, truth)) < BIAS_GATE


@strategies
def test_cosine_improves_with_population(strategy, problem):
    """Sanity: more members, better direction. If this fails nothing else means much."""
    _q, _theta, truth, _model = problem
    cos = [
        float(metrics.cosine_similarity(
            jnp.mean(run(strategy, problem, shp.centered, n=n, replicates=64), axis=0), truth))
        for n in (4, 32, 256)
    ]
    assert cos[0] < cos[1] < cos[2], cos
