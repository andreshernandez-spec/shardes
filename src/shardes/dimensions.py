"""Sampling dimension: how many independent numbers a perturbation actually draws.

This is the x-axis of figure F5, and the two panels measure different things on purpose.

Full rank draws one number per parameter, so the space it samples in *is* the space the
gradient lives in. Rank r draws `r*(m+n)` per matrix, so the perturbations still live in
R^(mn) while the randomness lives in something 256x smaller (at m=n=512, r=1, six
matrices: 1,572,864 against 6,144).

`d_eff` here is the second quantity, the dimension **sample design operates in**. That is
the right one for the coupling claim: coupling cannot help you span more of R^(mn), only
spread the a's and b's more evenly on their spheres, and its leverage scales like
N/d_sampling (docs/00-context.md). Under full rank the two coincide, which is why the
distinction never arises in the classical literature.

Say this in the F5 caption. A reader who assumes d_eff means gradient-space dimension will
think the panels are not comparable.
"""

import jax

from shardes.types import PyTree

FULL = "full"


def sampling_dimension(params: PyTree, rank: int | str) -> int:
    """Independent numbers drawn per member, summed over the whole params tree.

    One member's perturbation covers every leaf at once, so this sums rather than taking
    a per-leaf maximum or picking a representative matrix.

    Leaves with fewer than two axes (norm scales, biases) have no low-rank factorisation,
    so they contribute their full size at any rank. With the Phase 0 block every leaf is a
    matrix and that branch is unused, but it is what makes the number correct once mixed
    leaf types arrive in Phase 1.
    """
    total = 0
    for leaf in jax.tree.leaves(params):
        if rank == FULL or leaf.ndim < 2:
            total += int(leaf.size)
        else:
            total += int(rank) * int(sum(leaf.shape))
    return total
