"""Per-leaf step size. Shared by every strategy's `apply`.

`sigma` may be a scalar (isotropic ES, one global step size) or a pytree matching `params`
(a per-coordinate diagonal). Everything downstream multiplies it elementwise against a
params-shaped array, so the only thing that has to change is that the multiplication becomes
a three-argument `tree.map`.

**This is what a diagonal costs, and the reason Option 1 was dropped in docs/02 C1.4.**
`sample` produces *unit-scale* perturbations and `apply` scales them, so per-coordinate step
sizes are a type widening on an argument that already exists — no fourth protocol method, no
state threaded through `sample`, and it composes with `LowRank` because a scale is not a
distribution. A rank-one covariance term would couple coordinates and could not be expressed
this way, which is exactly the line between what ships and what does not.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shardes.types import Array, PyTree


class Separable(NamedTuple):
    """A per-coordinate sigma for a rank-2 leaf, written as the outer product `u v^T`.

    **The only per-coordinate family a structured weight can carry, and the reason is
    arithmetic rather than convenience.** `LowRank` perturbs a leaf as `W + sigma * (A B^T)`
    with `*` elementwise, and applies it as two GEMMs against the factors so the sum is never
    formed (invariant 3). A general elementwise sigma raises the rank of that product above
    `r`: measured at `m=5, k=4, r=2`, the rank goes to 4. There is then nothing left to apply.

    An outer product is exactly the case that survives:

        (u v^T) * (A B^T) == (u[:, None] * A) @ (v[:, None] * B).T

    still rank `r`, still two GEMMs, and `LowRank.apply` folds `u` and `v` into the factors
    rather than materialising anything. Verified in
    `tests/test_lowrank.py::test_a_separable_sigma_equals_its_dense_form`.

    This covers what a per-coordinate schedule usually means: `u` is a step size per output
    unit and `v` per input unit. What it cannot express is an arbitrary `(m, k)` schedule,
    which is the part that does not exist for a low-rank perturbation at all.

    `dense()` is for the consumers that genuinely need the `(m, k)` form. `tell` is one: the
    estimator divides the update by sigma elementwise, and the update is already `(m, k)` by
    then, so densifying costs nothing new there. The forward pass never calls it.
    """

    u: Array  # (m,), scales the output axis
    v: Array  # (k,), scales the input axis

    def dense(self) -> Array:
        return jnp.outer(self.u, self.v)


def is_separable(x) -> bool:
    """`is_leaf` predicate. `Separable` is a pytree so it crosses `jit`, which means every
    tree walk over a sigma has to be told to stop at it or it sees `u` and `v` as two
    separate leaves and pairs them against the wrong params."""
    return isinstance(x, Separable)


def densify(sigma: PyTree) -> PyTree:
    """Every `Separable` replaced by its `(m, k)` form, for consumers that need one array."""
    return jax.tree.map(lambda s: s.dense() if is_separable(s) else s, sigma,
                        is_leaf=is_separable)



def per_leaf(sigma: PyTree, like: PyTree) -> PyTree:
    """`sigma` broadcast to `like`'s structure.

    A scalar becomes the same scalar at every leaf; a tree already matching `like` passes
    through untouched. Deciding by tree structure rather than by `isinstance` keeps a traced
    scalar and a Python float on the same path, which matters because `sigma` lives in the
    ES state and therefore crosses `jit`.
    """
    if jax.tree.structure(sigma, is_leaf=is_separable) == jax.tree.structure(like):
        return sigma
    # **Only a leaf broadcasts.** This used to fall through to `tree.map` for anything whose
    # structure did not match, so a sigma tree with one wrong key was silently treated as a
    # scalar and the whole object was broadcast to every leaf. The run then looked normal.
    # A mismatched tree is a caller error and the only question is whether it is reported.
    # `treedef_is_leaf`, not `num_leaves == 1`: a one-key dict has a single leaf and is not
    # a scalar, and treating it as one is the exact silent broadcast being closed here.
    if not jax.tree_util.treedef_is_leaf(jax.tree.structure(sigma, is_leaf=is_separable)):
        raise ValueError(
            f"sigma has structure {jax.tree.structure(sigma)}, which is neither a scalar nor "
            f"a match for params' structure {jax.tree.structure(like)}. A per-coordinate "
            "sigma must have exactly the leaves params has; anything else was previously "
            "broadcast as though it were a scalar."
        )
    return jax.tree.map(lambda _: sigma, like)
