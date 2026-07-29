"""Fitness shaping.

The defining property of rank shaping is that it depends on the *ordering* of the
fitnesses and nothing else. That is what makes ES robust to reward scale and to outliers,
and it is the property that catches an implementation which quietly passes raw values
through: such an implementation still produces a sensible-looking update direction, so
the estimator tests do not notice.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import shaping

N = 16


@pytest.fixture
def fitness():
    return jax.random.normal(jax.random.key(0), (N,), dtype=jnp.float32)


def test_none_is_the_identity(fitness):
    assert jnp.array_equal(shaping.none(fitness), fitness)


@pytest.mark.parametrize(
    "monotone",
    [lambda f: 3 * f, lambda f: f + 100.0, lambda f: f**3, lambda f: jnp.exp(f)],
    ids=["scale", "shift", "cube", "exp"],
)
def test_centered_ranks_depend_only_on_ordering(fitness, monotone):
    """Any strictly increasing map of the fitnesses must give identical weights.

    This is the test that distinguishes a rank transform from a rescaled raw value. An
    implementation returning `fitness/(n-1) - 0.5` produces a plausible update direction
    and passes every estimator test; it fails here immediately.
    """
    assert jnp.allclose(
        shaping.centered_ranks(fitness), shaping.centered_ranks(monotone(fitness))
    )


def test_centered_ranks_are_permutation_equivariant(fitness):
    perm = jax.random.permutation(jax.random.key(1), N)
    got = shaping.centered_ranks(fitness[perm])
    assert jnp.allclose(got, shaping.centered_ranks(fitness)[perm])


def test_centered_ranks_survive_an_outlier(fitness):
    """One absurd fitness must not dominate. This is most of why ES uses ranks."""
    spiked = fitness.at[3].set(1e9)
    base = shaping.centered_ranks(fitness.at[3].set(fitness.max() + 1.0))
    assert jnp.allclose(shaping.centered_ranks(spiked), base)


def test_centered_ranks_span_the_unit_interval(fitness):
    w = shaping.centered_ranks(fitness)
    assert jnp.isclose(w.min(), -0.5)
    assert jnp.isclose(w.max(), 0.5)
    assert jnp.isclose(w.sum(), 0.0, atol=1e-5)


def test_centered_sums_to_zero(fitness):
    assert jnp.isclose(shaping.centered(fitness).sum(), 0.0, atol=1e-4)


def test_centered_carries_the_bias_correction(fitness):
    """The n/(n-1) factor is the whole point; without it the estimator targets
    (1 - 1/n) grad f. Checked against the naive version directly."""
    naive = fitness - jnp.mean(fitness)
    assert jnp.allclose(shaping.centered(fitness), naive * (N / (N - 1)), rtol=1e-6)
    assert not jnp.allclose(shaping.centered(fitness), naive)


def test_centered_is_shift_invariant(fitness):
    """A constant added to every fitness is not information."""
    assert jnp.allclose(shaping.centered(fitness + 1000.0), shaping.centered(fitness), atol=1e-3)


@pytest.mark.parametrize("name", ["none", "centered", "centered_ranks"])
def test_single_member_does_not_divide_by_zero(name):
    """n = 1 shows up in chunked runs with chunk=1; n-1 must not blow up."""
    out = shaping.BY_NAME[name](jnp.array([1.5], dtype=jnp.float32))
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


def test_by_name_covers_every_public_shaping():
    """A shaping reachable by config must be reachable by name, or a sweep silently
    cannot select it."""
    public = {n for n in dir(shaping) if not n.startswith("_") and callable(getattr(shaping, n))}
    public -= {"Array"}  # the type alias import
    assert public == set(shaping.BY_NAME), public ^ set(shaping.BY_NAME)
