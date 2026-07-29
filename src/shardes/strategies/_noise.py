"""Per-member Gaussian noise, shared by IIDGaussian and SeedRegenerated.

Shared on purpose. The two strategies differ only in *when* the noise is materialized:
IIDGaussian stores it, SeedRegenerated re-derives it. If they ever produced different
noise, Qiu et al.'s seed trick would have stopped reproducing full-rank noise and would be
a different algorithm. `test_seed_regenerated_matches_iid_gaussian` asserts they agree, and
this module is what makes that cheap to guarantee.
"""

import jax

from shardes.types import Array, Key, PyTree


def member_noise(base_key: Key, like: PyTree, member_id: Array) -> PyTree:
    """Unit-normal noise for one member, shaped and typed like `like`.

    Derived from fold_in(base_key, member_id), so it depends only on the member's own
    global id: not on how many members are drawn, not on position, not on the device.

    Splitting across leaves within a member is safe. The leaf count is a property of the
    params tree, not of the batch, so it cannot vary with how members are grouped.
    """
    leaves, treedef = jax.tree.flatten(like)
    keys = jax.random.split(jax.random.fold_in(base_key, member_id), len(leaves))
    return jax.tree.unflatten(
        treedef,
        [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)],
    )
