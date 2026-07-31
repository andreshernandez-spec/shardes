"""SeedRegenerated: Qiu et al. Full-rank noise stored as seeds, regenerated on the fly.

Mathematically identical to IIDGaussian. The difference is entirely one of schedule:
IIDGaussian materializes all n perturbations and keeps them; this stores a key and the
member ids and re-derives each member when it is needed, twice per generation (once to
evaluate, once to contract).

That is Qiu's bet made concrete. Storage drops from `n * |params|` to `|params|`, and the
price is that members cannot share a base GEMM, because each genuinely is a different
weight matrix. `docs/00-context.md` states the tradeoff; this file is the half of it that
gives up throughput to keep full-rank noise.

**The scan is the point.** vmapping the regeneration would materialize all n perturbations
again and buy nothing. `lax.scan` accumulates into one params-shaped buffer, so peak memory
is O(|params|) whatever n is. If a profile shows an (n, ...) array here, the implementation
has quietly become IIDGaussian with extra steps.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from shardes.coupling import GAUSSIAN, Coupling
from shardes.strategies._noise import member_noise
from shardes.strategies._scale import per_leaf
from shardes.types import Array, Key, PyTree


class SeedPerturbation(NamedTuple):
    """base_key and member_ids are the regeneration state. There is no eps field.

    `like` is the params tree, held by reference for its shapes and dtypes. `contract`
    does not receive `params`, so the shapes have to travel with the perturbation, and a
    reference costs nothing: it is the same arrays that are already resident, so storage
    stays O(1) in n, which is the property this strategy exists for.

    The tidier alternative is a custom pytree with ShapeDtypeStruct in aux_data. It is
    more machinery for the same result; revisit if `like` ever gets mistaken for state.
    """

    base_key: Key
    member_ids: Array
    like: PyTree


class SeedRegenerated:
    """Full-rank noise regenerated from per-member seeds. Qiu et al.

    Coupling composes with regeneration for free: a coupled draw is still a pure function
    of (stream, member_id), so there is nothing extra to store and nothing to synchronize.
    """

    def __init__(self, coupling: Coupling = GAUSSIAN):
        self.coupling = coupling

    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> SeedPerturbation:
        """Does no work. That is the whole idea: the noise is a function of (key, id)."""
        return SeedPerturbation(base_key, member_ids, params)

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        params: PyTree,
        pert: SeedPerturbation,
        sigma: float,
    ) -> Callable[[Array], Array]:
        """One member at a time, perturb / evaluate / discard.

        A scan rather than a vmap, for the same reason as `contract`: members are
        different weight matrices, so there is no shared GEMM to win and nothing to gain
        from holding them all at once.
        """

        scale = per_leaf(sigma, params)

        def g(x: Array) -> Array:
            def step(carry, i):
                eps = member_noise(pert.base_key, pert.like, i, self.coupling)
                perturbed = jax.tree.map(lambda p, s, e: p + s * e, params, scale, eps)
                return carry, model(perturbed, x)

            _, out = jax.lax.scan(step, None, pert.member_ids)
            return out

        return g

    def contract(self, pert: SeedPerturbation, weights: Array) -> PyTree:
        """sum_i weights[i] * eps_i, regenerated one member at a time.

        Accumulates in f32 and never holds more than one member's noise plus the running
        sum. No division by n_local: partial contractions must sum to the whole.
        """
        w = weights.astype(jnp.float32)

        def step(acc, iw):
            i, wi = iw
            eps = member_noise(pert.base_key, pert.like, i, self.coupling)
            return jax.tree.map(lambda a, e: a + wi * e.astype(jnp.float32), acc, eps), None

        # zeros_like, not zeros(x.shape): the accumulator has to inherit whatever type
        # metadata the leaf carries, and a static shape discards it. Under shard_map the
        # scan body produces values varying across the manual axis, so an accumulator built
        # from a bare shape starts invariant and `lax.scan` rejects the carry as changing
        # type mid-loop. Outside shard_map the two are identical, which is why this reads
        # like a style preference and is not one. contraction.contract_sharded is the other
        # half; neither works alone.
        init = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), pert.like)
        acc, _ = jax.lax.scan(step, init, (pert.member_ids, w))
        return acc
