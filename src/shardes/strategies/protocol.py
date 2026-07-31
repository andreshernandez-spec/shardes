"""The PerturbationStrategy protocol: sample / apply / contract.

Settled in docs/01-phase0-estimator-harness.md C0.1, which carries the reasoning. This
file is the spec; the implementations live beside it.

Protocols, not ABCs. Structural typing keeps user-defined strategies first-class without
inheritance (docs/conventions.md).

The perturbation scheme cannot be a parameter, because it determines how the forward pass
is structured. That is why it owns all three steps (docs/00-context.md).
"""

from typing import Callable, Protocol, runtime_checkable

from shardes.types import Array, Key, PyTree


@runtime_checkable
class Perturbation(Protocol):
    """Whatever `sample` returns, plus enough state to regenerate itself.

    Regenerability is a requirement, not an implementation detail. It is what lets one
    type serve both contraction strategies: A re-derives all N members from
    (base_key, member_ids) on every device, B keeps its local shard and psums. Strategies
    may cache materialized factors alongside these two fields.
    """

    base_key: Key
    member_ids: Array


@runtime_checkable
class PerturbationStrategy(Protocol):
    """sample / apply / contract. Unit scale throughout; sigma belongs to the ES state."""

    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> Perturbation:
        """Unit-scale perturbation for exactly the members in `member_ids`.

        member_ids: (n_local,) int32 of GLOBAL member indices.

        Member i derives from fold_in(base_key, i) and depends only on
        (base_key, i, params shapes). Never on n_local, on where i sits in the array, on
        the device, or on any counter. Declaring member_ids as P("pop") is what makes
        this work unchanged inside and outside a shard_map, and it is why there is
        nowhere here to put a device index.

        Leaves keep their (m, n) structure. No ravel_pytree.
        """
        ...

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        params: PyTree,
        pert: Perturbation,
        sigma: float | PyTree,
    ) -> Callable[[Array], Array]:
        """Given model(params, x) -> y, return g(x) -> (n_local, ...) over all members.

        `sigma` is a scalar (isotropic) or a pytree matching `params` (a per-coordinate
        diagonal). `strategies._scale.per_leaf` normalizes the two, and every implementation
        should use it rather than branching: the multiplication is elementwise either way.

        Full rank materializes per member, or regenerates from seed. Low rank rewrites
        x @ W.T into x @ W.T + (x @ B) @ A.T and never materializes.

        sigma enters here because the forward pass needs it, and nowhere else in this
        protocol. How LowRank reaches the model's matmuls is deferred; see docs/01 C0.1.
        """
        ...

    def contract(self, pert: Perturbation, weights: Array) -> PyTree:
        """sum_i weights[i] * eps_i, params-shaped, unit scale.

        weights: (n_local,) shaped fitness. The result must not depend on sigma; the
        1/(N sigma) factor is applied by tell.

        Accumulate in f32 even when the perturbation is bf16: summing 2^18 bf16 terms
        loses several digits. Partial contractions over disjoint member_ids must sum to
        the whole, which is what makes both chunking and Strategy B work.

        The only place a full (m, n) tensor is instantiated.
        """
        ...
