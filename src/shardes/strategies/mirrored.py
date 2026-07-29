"""Mirrored(inner): antithetic pairs. Member 2k is +eps_k, member 2k+1 is -eps_k.

A wrapper, not a fourth strategy, and not optional either. Mirrored sampling is standard
in ES, so it is the honest baseline; a win measured against unmirrored i.i.d. is a win
against a strawman, and much of the easy variance reduction is already spent by the time
coupling gets a turn (docs/00-context.md, obstacle 1).

**Member ids must arrive as complete adjacent pairs**: `2k` immediately followed by
`2k+1`. That is a real constraint on callers, not an implementation convenience.

The alternative was deriving the pair and sign from each id independently, which tolerates
any batch but costs 4x: `inner.sample` would generate n perturbations where only n/2 are
distinct, and `apply` would have to evaluate every member at both +sigma and -sigma and
select. Pairing by position keeps the work at n/2 distinct draws and n evaluations.

Consequences worth knowing before they bite:

- a chunk size must be even, or a chunk splits a pair and the antithetic cancellation is
  lost. `sample` raises on an odd count, so this fails loudly.
- sharding must split on an even boundary, for the same reason. With population a power
  of two and device counts a power of two, it always does.
- under uniform weights the contraction is **exactly zero**, since the pair contributions
  cancel. That is correct and is the point: constant fitness means no update.
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp

from shardes.strategies.protocol import PerturbationStrategy
from shardes.types import Array, Key, PyTree


class MirroredPerturbation(NamedTuple):
    """`inner` covers n/2 distinct directions; the n members are their +/- images."""

    base_key: Key
    member_ids: Array
    inner: PyTree


class Mirrored:
    def __init__(self, inner: PerturbationStrategy):
        self.inner = inner

    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> MirroredPerturbation:
        n = int(member_ids.shape[0])
        if n % 2:
            raise ValueError(
                f"Mirrored needs an even member count, got {n}. Members pair as "
                "(2k, 2k+1), so an odd count or an odd chunk splits a pair and loses the "
                "antithetic cancellation."
            )
        # `// 2` on the even ids, so the direction index comes from the id rather than
        # from the position in the batch. Member 7 is the negative image of direction 3
        # whatever batch it arrives in, which is what the seed contract requires.
        base_ids = member_ids[0::2] // 2
        return MirroredPerturbation(
            base_key, member_ids, self.inner.sample(base_key, params, base_ids)
        )

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        params: PyTree,
        pert: MirroredPerturbation,
        sigma: float,
    ) -> Callable[[Array], Array]:
        """Two inner passes at +sigma and -sigma, interleaved.

        Negating sigma rather than the perturbation is what keeps this a wrapper: the
        inner perturbation stays opaque, and every strategy already handles a signed
        sigma because it only ever multiplies.
        """
        positive = self.inner.apply(model, params, pert.inner, sigma)
        negative = self.inner.apply(model, params, pert.inner, -sigma)

        def g(x: Array) -> Array:
            plus, minus = positive(x), negative(x)
            return jnp.stack([plus, minus], axis=1).reshape((-1,) + plus.shape[1:])

        return g

    def contract(self, pert: MirroredPerturbation, weights: Array) -> PyTree:
        """sum_k (w_2k - w_{2k+1}) eps_k.

        The signs fold into the weights because `contract` is linear, so no strategy needs
        to know how to negate its own perturbation.
        """
        return self.inner.contract(pert.inner, weights[0::2] - weights[1::2])
