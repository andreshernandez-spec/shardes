"""IIDGaussian: full-rank unstructured Gaussian noise, materialized. Textbook OpenAI-ES.

Slow and memory-hungry on purpose. It is the reference the other two strategies are
checked against, and it is the naive baseline both papers beat (E9).

Storage is `n_local * |params| * 4` bytes, which is why this is the reference and not the
workhorse: at N = 2^18 with m = n = 512 that is 275 TB. `SeedRegenerated` exists to avoid
exactly this, and differs from this file by dropping one field.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from shardes.types import Array, Key, PyTree


class IIDPerturbation(NamedTuple):
    """base_key and member_ids are the regeneration state the protocol requires.

    eps mirrors the params tree, each leaf shaped (n_local, *leaf.shape), unit scale.

    A NamedTuple rather than a frozen dataclass: this crosses jit, vmap and shard_map
    boundaries, and a NamedTuple is a pytree without needing registration.
    """

    base_key: Key
    member_ids: Array
    eps: PyTree


class IIDGaussian:
    """Full-rank i.i.d. normal perturbation, materialized per member."""

    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> IIDPerturbation:
        """Unit-scale noise for exactly the members in `member_ids`.

        member_ids: (n_local,) int, GLOBAL member indices.

        Member i derives from fold_in(base_key, i), so it depends only on its own id:
        not on n_local, not on where it sits in the array, not on the device. That is
        also what lets a shard derive its own members without knowing N at all, which
        `jax.random.split(base_key, N)` cannot do without building all N keys first.

        Splitting across *leaves* within a member is fine. The leaf count is a property
        of params, not of the batch.
        """
        leaves, treedef = jax.tree.flatten(params)

        def one(i: Array) -> PyTree:
            keys = jax.random.split(jax.random.fold_in(base_key, i), len(leaves))
            return jax.tree.unflatten(
                treedef,
                [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)],
            )

        return IIDPerturbation(base_key, member_ids, jax.vmap(one)(member_ids))

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        params: PyTree,
        pert: IIDPerturbation,
        sigma: float,
    ) -> Callable[[Array], Array]:
        """model(params, x) -> y becomes g(x) -> (n_local, ...) over every member.

        Materializing the perturbed params per member is the whole cost of this strategy
        and the reason the low-rank path exists. It is also what makes this the reference
        the others are checked against, so keep it obvious rather than clever.
        """

        def g(x: Array) -> Array:
            def one(eps: PyTree) -> Array:
                perturbed = jax.tree.map(lambda p, e: p + sigma * e, params, eps)
                return model(perturbed, x)

            return jax.vmap(one)(pert.eps)

        return g

    def contract(self, pert: IIDPerturbation, weights: Array) -> PyTree:
        """sum_i weights[i] * eps_i, params-shaped, unit scale.

        No division by n_local. Partial contractions over disjoint members have to sum to
        the whole, which is what makes chunking, contraction Strategy B, and streaming
        full rank at large N all valid. The 1/(N sigma) factor belongs to tell.

        Accumulated and returned in f32 even when the perturbation is bf16; the caller
        downcasts if it wants to.
        """
        w = weights.astype(jnp.float32)
        return jax.tree.map(
            lambda e: jnp.einsum("n,n...->...", w, e.astype(jnp.float32)), pert.eps
        )
