"""FWHT against the exact oracle that ships in JAX.

`jax.scipy.linalg.hadamard(n)` is a dense O(n^2) Sylvester constructor: useless as an
implementation, perfect as a test (docs/01-phase0-estimator-harness.md C0.3).

Tolerance: comparisons are norm-relative rather than elementwise. Both the transform and
the dense oracle accumulate n terms in f32, and individual output entries pass through
zero, where an elementwise rtol is meaningless. rtol=1e-6 is the f32 exact-oracle figure
from docs/conventions.md.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes.transforms.fwht import fwht

RTOL = 1e-6
SIZES = [1, 2, 4, 8, 16, 64, 256, 1024, 4096]


def rel_err(got, want):
    return jnp.linalg.norm(got - want) / jnp.linalg.norm(want)


def dense_oracle(n):
    """The O(n^2) Sylvester Hadamard matrix, cast to f32. Ships as int32."""
    return jax.scipy.linalg.hadamard(n).astype(jnp.float32)


@pytest.mark.parametrize("n", SIZES)
def test_matches_dense_oracle(n):
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    assert rel_err(fwht(x), dense_oracle(n) @ x) < RTOL


@pytest.mark.parametrize("n", SIZES)
def test_involution(n):
    """fwht is its own inverse up to a factor of n. No normalization is applied."""
    x = jax.random.normal(jax.random.key(1), (n,), dtype=jnp.float32)
    assert rel_err(fwht(fwht(x)), n * x) < RTOL


@pytest.mark.parametrize("n", [4, 16, 256])
def test_basis_vector_gives_column(n):
    """fwht(e_i) is column i of H.

    Catches the ordering bug that matters: Sylvester (natural) order versus sequency
    (Walsh) order. Both are called "the" Hadamard transform and they differ by a
    bit-reversal-and-Gray-code permutation of the rows, so a sequency-ordered
    implementation still passes the involution test.
    """
    H = dense_oracle(n)
    eye = jnp.eye(n, dtype=jnp.float32)
    for i in range(n):
        assert rel_err(fwht(eye[i]), H[:, i]) < RTOL


def test_batched_matches_loop():
    """Leading batch dims are independent transforms along the last axis."""
    n, batch = 64, (3, 5)
    x = jax.random.normal(jax.random.key(2), (*batch, n), dtype=jnp.float32)
    got = fwht(x)
    assert got.shape == x.shape
    for i in range(batch[0]):
        for j in range(batch[1]):
            assert rel_err(got[i, j], fwht(x[i, j])) < RTOL


def test_linearity():
    n = 128
    ka, kb = jax.random.split(jax.random.key(3))
    x = jax.random.normal(ka, (n,), dtype=jnp.float32)
    y = jax.random.normal(kb, (n,), dtype=jnp.float32)
    a, b = 2.5, -0.75
    assert rel_err(fwht(a * x + b * y), a * fwht(x) + b * fwht(y)) < RTOL


def test_jit_and_vmap():
    """Has to survive both, since it runs inside the sampling path."""
    n = 32
    x = jax.random.normal(jax.random.key(4), (7, n), dtype=jnp.float32)
    assert rel_err(jax.jit(fwht)(x), fwht(x)) < RTOL
    assert rel_err(jax.vmap(fwht)(x), fwht(x)) < RTOL


def test_dtype_preserved():
    n = 16
    for dtype in (jnp.float32, jnp.bfloat16):
        x = jax.random.normal(jax.random.key(5), (n,), dtype=dtype)
        assert fwht(x).dtype == dtype


@pytest.mark.parametrize("n", [0, 3, 5, 6, 12, 100])
def test_rejects_non_power_of_two(n):
    x = jnp.ones((n,), dtype=jnp.float32)
    with pytest.raises(ValueError):
        fwht(x)
