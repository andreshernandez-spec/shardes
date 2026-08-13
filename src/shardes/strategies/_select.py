"""`split`: a perturbation reshaped so `ShardedES.apply` can vmap one row per device.

`ShardedES.apply` divides the population by vmapping over `n_rows` device rows. Each row needs
the sub-perturbation for members `[row*n/n_rows, (row+1)*n/n_rows)`, and only the strategy
knows which of its arrays carry a member axis: `SeedPerturbation.like` is the params tree and
carries none, `MirroredPerturbation.inner` carries `n/2` directions rather than `n`.

**The perturbation is mapped, not closed over, and that distinction is the whole design.**
An earlier attempt closed over the whole perturbation and sliced it per row with
`dynamic_slice`. It is numerically identical and it does not shard: a closed-over array has
to exist whole on every device before it can be sliced, so each device materialised all `n`
members. Measured, `iid_gaussian` at `D=8` went to **5.55x** the per-device FLOPs of the
sweep commit, which is the replicated-evaluation defect again with the perturbation standing
in for the evaluation. Mapping the array is what makes GSPMD partition it.

So `split` returns the perturbation reshaped to a leading `(n_rows, ...)` axis *and* the
`in_axes` tree that says which leaves carry it. Both are needed: `in_axes=0` everywhere would
demand tiling `like` and `base_key` `n_rows` times, and `like` is the model.
"""

from __future__ import annotations

import jax

from shardes.types import Array, PyTree


def rows(x: Array, n_rows: int) -> Array:
    """`(n, ...)` to `(n_rows, n // n_rows, ...)`, row-major.

    Row `k` is members `[k*n/n_rows, (k+1)*n/n_rows)`, the same contiguous split
    `sharding.member_ids` hands the mesh. They have to agree, or a member is evaluated under
    one device's shard and contracted under another's.
    """
    return x.reshape(n_rows, x.shape[0] // n_rows, *x.shape[1:])


def rows_like(tree: PyTree, n_rows: int) -> PyTree:
    """`rows` at every leaf, for a subtree whose leaves all carry the member axis."""
    return jax.tree.map(lambda x: rows(x, n_rows), tree)


def mapped(tree: PyTree) -> PyTree:
    """An `in_axes` tree of 0 at every leaf. For the member-axis half of a perturbation."""
    return jax.tree.map(lambda _: 0, tree)


def unmapped(tree: PyTree) -> PyTree:
    """An `in_axes` tree of None at every leaf. For everything with no member axis."""
    return jax.tree.map(lambda _: None, tree)
