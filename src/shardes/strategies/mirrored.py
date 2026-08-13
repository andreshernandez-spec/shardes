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

import jax
import numpy as np
import jax.numpy as jnp

from shardes.strategies.protocol import PerturbationStrategy
from shardes.strategies._select import mapped, rows, rows_like, unmapped
from shardes.types import Array, Key, PyTree


class MirroredPerturbation(NamedTuple):
    """`inner` covers n/2 distinct directions; the n members are their +/- images."""

    base_key: Key
    member_ids: Array
    inner: PyTree


def _check_alignment(member_ids: Array) -> None:
    """Refuse a batch that is not whole (2k, 2k+1) pairs, when that is knowable.

    **Only checkable on concrete ids, and that is a real limitation rather than laziness.**
    Under `jit` `member_ids` is a tracer and its values do not exist yet, so
    `if (ids[0::2] % 2).any(): raise` cannot be written. The structural guarantee is
    `check_population`, which runs at construction where `n` and the device count are static;
    this catches the direct caller, the test, and the notebook, which is where a misaligned
    batch would actually come from.

    A pair split across a batch boundary does not raise anywhere else and does not produce a
    wrong shape. It produces a population whose antithetic cancellation is quietly gone, and
    members whose sign depends on how they were batched.
    """
    try:
        ids = np.asarray(member_ids)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return  # traced, so the values are not available; check_population covers this
    if ids.size % 2 or (ids[0::2] % 2 != 0).any() or (ids[1::2] != ids[0::2] + 1).any():
        raise ValueError(
            f"Mirrored needs whole adjacent pairs starting on an even id, got {ids.tolist()}. "
            "Members pair as (2k, 2k+1) and the direction is read from the pair's position, "
            "so a batch that starts mid-pair gives a member the opposite sign to the one it "
            "has in any other batch, which breaks the seed contract in sharding.py."
        )


class Mirrored:
    #: Members per indivisible group. `ShardedES` reads this instead of asking
    #: `isinstance(strategy, Mirrored)`, so a user-defined paired strategy can ask for the
    #: same alignment without being this class. The protocol treats a strategy with no
    #: `pairing` attribute as 1, which is every unpaired strategy.
    pairing = 2

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
        # **The slice is positional, and it is only correct because the batch is aligned.**
        # This comment used to claim the direction came from the id rather than the
        # position, which is false and was measured: with ids [0, 1] member 1 is the
        # negative image of direction 0, and with ids [1, 2] the same member 1 is the
        # positive image of direction 0. The same global member, two perturbations.
        #
        # The precondition is that the batch is a whole number of adjacent (2k, 2k+1) pairs
        # starting on an even id. `ShardedES` guarantees it: `member_ids` is contiguous and
        # `check_population` refuses a population whose per-device count is not a multiple
        # of `pairing`, so every shard starts on an even id. `_check_alignment` re-checks it
        # for anyone calling the strategy directly.
        _check_alignment(member_ids)
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
        # Negated per leaf, so a per-coordinate diagonal mirrors the same way a scalar
        # does. `jax.tree.map` over a bare scalar is the identity path, so this is not a
        # special case for the isotropic caller.
        positive = self.inner.apply(model, params, pert.inner, sigma)
        negative = self.inner.apply(model, params, pert.inner,
                                    jax.tree.map(lambda s: -s, sigma))

        def g(x: Array) -> Array:
            plus, minus = positive(x), negative(x)
            return jnp.stack([plus, minus], axis=1).reshape((-1,) + plus.shape[1:])

        return g

    def split(self, pert: MirroredPerturbation, n_rows: int):
        """Recursive, because `inner` covers `n/2` directions rather than `n` members.

        Pairs are adjacent and the split is contiguous, so row `k`'s members
        `[k*n/D, (k+1)*n/D)` are exactly directions `[k*n/(2D), (k+1)*n/(2D))`. Delegating to
        the inner's own `split` gets that without this class knowing the inner's layout, the
        same reason `apply` and `contract` delegate.
        """
        inner, inner_axes = self.inner.split(pert.inner, n_rows)
        return (
            MirroredPerturbation(pert.base_key, rows(pert.member_ids, n_rows), inner),
            MirroredPerturbation(None, 0, inner_axes),
        )

    def contract(self, pert: MirroredPerturbation, weights: Array) -> PyTree:
        """sum_k (w_2k - w_{2k+1}) eps_k.

        The signs fold into the weights because `contract` is linear, so no strategy needs
        to know how to negate its own perturbation.
        """
        return self.inner.contract(pert.inner, weights[0::2] - weights[1::2])
