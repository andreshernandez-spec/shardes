"""Test objectives, checked against their own oracles.

The quadratic is the base of the whole Phase 0 measurement: every estimator claim is
relative to its analytic gradient. If `grad` is wrong, nothing downstream means anything,
so it gets checked against autodiff, which is a genuinely independent implementation with
different failure modes.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes.problems import quadratic

RTOL = 1e-6
DIMS = [1, 2, 8, 64]


def rel_err(got, want):
    return jnp.linalg.norm(got - want) / jnp.linalg.norm(want)


@pytest.fixture
def q():
    return quadratic.make(jax.random.key(0), 32, condition_number=100.0)


@pytest.mark.parametrize("d", DIMS)
def test_grad_matches_autodiff(d):
    """The analytic gradient against backprop. Two independent derivations."""
    qd = quadratic.make(jax.random.key(1), d)
    theta = jax.random.normal(jax.random.key(2), (d,), dtype=jnp.float32)
    autodiff = jax.grad(lambda t: quadratic.value(qd, t))(theta)
    assert rel_err(quadratic.grad(qd, theta), autodiff) < RTOL


def test_symmetric(q):
    assert jnp.array_equal(q.H, q.H.T)


def test_positive_definite(q):
    assert jnp.min(jnp.linalg.eigvalsh(q.H)) > 0


@pytest.mark.parametrize("cond", [1.0, 10.0, 1000.0])
def test_spectrum_is_what_was_asked_for(cond):
    """Conditioning is chosen, not discovered, so assert it rather than trusting it."""
    d = 64
    qd = quadratic.make(jax.random.key(3), d, condition_number=cond)
    eigs = jnp.sort(jnp.linalg.eigvalsh(qd.H))
    want = jnp.logspace(0.0, jnp.log10(cond), d)
    assert rel_err(eigs, want) < 1e-4  # eigvalsh on f32, looser than the oracle figure
    assert jnp.isclose(eigs[-1] / eigs[0], cond, rtol=1e-3)


def test_value_is_quadratic(q):
    """f(a*theta) == a^2 f(theta), and f(0) == 0. Catches a stray factor of two."""
    theta = jax.random.normal(jax.random.key(4), (32,), dtype=jnp.float32)
    assert jnp.isclose(quadratic.value(q, jnp.zeros_like(theta)), 0.0)
    for a in (2.0, -3.0, 0.5):
        got = quadratic.value(q, a * theta)
        assert rel_err(got, a**2 * quadratic.value(q, theta)) < RTOL


def test_grad_is_zero_at_the_minimum(q):
    assert jnp.linalg.norm(quadratic.grad(q, jnp.zeros(32))) == 0.0


def test_deterministic():
    """Same key, same H. An experiment you cannot re-run is not a result."""
    a = quadratic.make(jax.random.key(5), 16)
    b = quadratic.make(jax.random.key(5), 16)
    assert jnp.array_equal(a.H, b.H)


def test_different_keys_differ():
    a = quadratic.make(jax.random.key(6), 16)
    b = quadratic.make(jax.random.key(7), 16)
    assert not jnp.allclose(a.H, b.H)


def test_jit_and_grad_compose(q):
    theta = jax.random.normal(jax.random.key(8), (32,), dtype=jnp.float32)
    assert jnp.isclose(jax.jit(quadratic.value)(q, theta), quadratic.value(q, theta))
    assert rel_err(jax.jit(quadratic.grad)(q, theta), quadratic.grad(q, theta)) < RTOL
