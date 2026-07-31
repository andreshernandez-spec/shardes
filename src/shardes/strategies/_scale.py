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

import jax

from shardes.types import PyTree


def per_leaf(sigma: PyTree, like: PyTree) -> PyTree:
    """`sigma` broadcast to `like`'s structure.

    A scalar becomes the same scalar at every leaf; a tree already matching `like` passes
    through untouched. Deciding by tree structure rather than by `isinstance` keeps a traced
    scalar and a Python float on the same path, which matters because `sigma` lives in the
    ES state and therefore crosses `jit`.
    """
    if jax.tree.structure(sigma) == jax.tree.structure(like):
        return sigma
    return jax.tree.map(lambda _: sigma, like)
