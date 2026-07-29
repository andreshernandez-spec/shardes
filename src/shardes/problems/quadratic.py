"""f(theta) = 0.5 theta^T H theta with known H. Gradient is H theta, analytic.

No autodiff involved. Catches sign errors, scaling errors and unbiasedness bugs, and it
should be the first thing that passes (docs/01-phase0-estimator-harness.md C0.4).

theta is a plain (d,) array here. Pytree structure arrives with the MLP; keeping this one
flat means a failure points at the estimator rather than at tree handling.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shardes.types import Array, Key


class Quadratic(NamedTuple):
    """H: (d, d), symmetric positive definite."""

    H: Array


def make(
    key: Key,
    d: int,
    *,
    condition_number: float = 100.0,
    dtype=jnp.float32,
) -> Quadratic:
    """Random symmetric positive-definite H with an exactly known spectrum.

    Eigenvalues are log-spaced over [1, condition_number], so conditioning is chosen rather
    than whatever a random draw happened to produce. Eigenvectors come from the QR of a
    Gaussian, which keeps them off the coordinate axes: an axis-aligned H would flatter any
    perturbation scheme that is itself axis-aligned.
    """
    g = jax.random.normal(key, (d, d), dtype=dtype)
    q, r = jnp.linalg.qr(g)

    # QR is only unique up to column signs. Pin R's diagonal positive so `make` gives the
    # same H on CPU and GPU, which device-invariance work later depends on.
    diag = jnp.diag(r)
    q = q * jnp.where(diag == 0, 1.0, jnp.sign(diag))

    eigs = jnp.logspace(0.0, jnp.log10(condition_number), d, dtype=dtype)
    h = (q * eigs) @ q.T
    return Quadratic(H=0.5 * (h + h.T))  # kill the asymmetry rounding leaves behind


def value(q: Quadratic, theta: Array) -> Array:
    """Scalar. theta: (d,)."""
    return 0.5 * theta @ q.H @ theta


def grad(q: Quadratic, theta: Array) -> Array:
    """Exact gradient H theta, (d,).

    Deliberately not autodiff. This is the oracle, so it has to be derived independently
    of the thing it checks (docs/conventions.md, "Numerics").
    """
    return q.H @ theta
