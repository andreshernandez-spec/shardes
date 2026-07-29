"""Per-member noise, shared by IIDGaussian and SeedRegenerated.

Shared on purpose. The two strategies differ only in *when* the noise is materialized:
IIDGaussian stores it, SeedRegenerated re-derives it. If they ever produced different
noise, Qiu et al.'s seed trick would have stopped reproducing full-rank noise and would be
a different algorithm. `test_seed_regenerated_matches_iid_gaussian` asserts they agree, and
this module is what makes that cheap to guarantee.

**The key derivation splits by leaf before folding in the member, not after.** A coupling
designs a point set across members, so it needs a `stream` key that every member of a
family shares; folding the member in first would give each member its own key and leave
nothing to design against. Both orders are index-stable, so this costs nothing and is what
makes `shardes.coupling` expressible at all.
"""

import jax

from shardes.coupling import GAUSSIAN, Coupling
from shardes.types import Array, Key, PyTree


def leaf_streams(base_key: Key, n_leaves: int) -> list[Key]:
    """One independent stream per leaf, independent of the member.

    The leaf count is a property of the params tree, not of the batch, so this cannot vary
    with how members are grouped.
    """
    return list(jax.random.split(base_key, n_leaves))


def member_noise(
    base_key: Key, like: PyTree, member_id: Array, coupling: Coupling = GAUSSIAN
) -> PyTree:
    """Unit-scale noise for one member, shaped and typed like `like`.

    Depends only on the member's own global id: not on how many members are drawn, not on
    position, not on the device.

    A leaf is flattened to draw and reshaped back. That is not the global flattening
    invariant 1 bans: it is per leaf, it never crosses a leaf boundary, and the (m, n)
    shape is restored before the leaf leaves this function. It matters because the design
    dimension under full rank *is* the whole leaf.
    """
    leaves, treedef = jax.tree.flatten(like)
    streams = leaf_streams(base_key, len(leaves))
    return jax.tree.unflatten(
        treedef,
        [
            coupling(s, member_id, x.size, x.dtype).reshape(x.shape)
            for s, x in zip(streams, leaves)
        ],
    )
