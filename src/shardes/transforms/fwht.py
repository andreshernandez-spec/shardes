"""Fast Walsh-Hadamard transform. O(n log n) butterfly.

Needed by Coupled(kind="orthogonal_hd"): the scalable orthogonal construction is
HD1 HD2 D3 ..., products of Hadamard transforms and Rademacher diagonals.

The oracle ships in JAX: jax.scipy.linalg.hadamard(n) @ x. It is a dense O(n^2) Sylvester
constructor, so it is useless as an implementation and perfect as a test.

A reference butterfly is enough. A Mosaic GPU kernel is a separate project with three
independent consumers (quantization rotations, SRHT, orthogonal ES) and is out of scope
here (docs/04-phase3-coupling.md, "adjacent things").
"""

import jax.numpy as jnp

from shardes.types import Array


def fwht(x: Array) -> Array:
    """Walsh-Hadamard transform along the last axis, Sylvester (natural) order.

    x: (..., n), n a power of two. Returns (..., n), same dtype.

    Unnormalized, so fwht(x) == hadamard(n) @ x exactly and fwht(fwht(x)) == n * x.
    Normalization is the caller's business: the HD construction wants 1/sqrt(n) per
    factor, and folding it in here would cost the exact oracle comparison.

    O(n log n). The loop runs over static shapes, so it unrolls at trace time into
    log2(n) stages and there is no Python left after jit.
    """
    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError(f"fwht needs a power-of-two last axis, got {n}")

    batch = x.shape[:-1]
    h = 1
    while h < n:
        # Flat index j splits as block*(2h) + half*h + pos, so half=0 and half=1 are the
        # butterfly partners j and j+h.
        y = x.reshape(*batch, n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = jnp.stack([a + b, a - b], axis=-2).reshape(*batch, n)
        h *= 2
    return x
