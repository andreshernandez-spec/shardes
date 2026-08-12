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
    # **Only a leaf broadcasts.** This used to fall through to `tree.map` for anything whose
    # structure did not match, so a sigma tree with one wrong key was silently treated as a
    # scalar and the whole object was broadcast to every leaf. The run then looked normal.
    # A mismatched tree is a caller error and the only question is whether it is reported.
    # `treedef_is_leaf`, not `num_leaves == 1`: a one-key dict has a single leaf and is not
    # a scalar, and treating it as one is the exact silent broadcast being closed here.
    if not jax.tree_util.treedef_is_leaf(jax.tree.structure(sigma)):
        raise ValueError(
            f"sigma has structure {jax.tree.structure(sigma)}, which is neither a scalar nor "
            f"a match for params' structure {jax.tree.structure(like)}. A per-coordinate "
            "sigma must have exactly the leaves params has; anything else was previously "
            "broadcast as though it were a scalar."
        )
    return jax.tree.map(lambda _: sigma, like)
