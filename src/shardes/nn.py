"""The linear seam a structured perturbation rewrites.

`dense(x, w)` is `x @ w.T` when `w` is an array, and defers to the weight itself when it
is a structured one. That single indirection is what makes `LowRank` expressible without
a jaxpr interpreter: the strategy substitutes structured leaves into the params tree, the
model calls `dense` exactly as before, and nothing else changes.

The cost is real and was accepted deliberately (docs/01 C0.1): a model has to be written
against `dense` rather than `x @ w.T`, so an arbitrary Flax module cannot be dropped in.
Arbitrary-model support is deferred. If the jaxpr route later proves tractable, `dense`
degrades to optional sugar rather than becoming a wrong turn.

`IIDGaussian` and `SeedRegenerated` substitute ordinary perturbed arrays and take the
array branch, so nothing about them changes.

There are two seams, not one, because there are two ways a weight gets used. `dense` covers
`x @ W.T`; `embed` covers `W[ids]`. An embedding is a gather rather than a matmul, which is
the whole reason it needed its own entry point and the reason EGGROLL's reference
implementation raises `NotImplementedError` there.
"""

from typing import Protocol, runtime_checkable

from shardes.types import Array


@runtime_checkable
class StructuredWeight(Protocol):
    """A weight that knows how to apply itself, instead of being a matrix.

    Deliberately not "low rank". The seam should not name one factorisation: a Kronecker
    or block-diagonal weight would implement the same method, and `dense` should not care.

    Implementations must be pytrees, since they travel through jit, vmap and shard_map
    inside the params tree.
    """

    def apply_to(self, x: Array) -> Array:
        """Return the equivalent of `x @ W.T` for this weight."""
        ...


@runtime_checkable
class GatherableWeight(Protocol):
    """A structured weight that can be indexed by row without being formed.

    Separate from `StructuredWeight` rather than a second method on it, so a weight that only
    knows how to be multiplied stays valid. `embed` says so explicitly when it meets one.
    """

    def gather(self, ids: Array) -> Array:
        """Return the equivalent of `W[ids]` for this weight."""
        ...


def embed(table: Array | GatherableWeight, ids: Array) -> Array:
    """table: (V, d) or structured, ids: (...,) int. Returns (..., d).

    **The seam that makes embeddings expressible under low-rank perturbation**, and the
    reason is a one-line identity rather than an algorithm:

        (E + s A B^T)[ids] = E[ids] + s * A[ids] @ B^T

    so the correction is a `(..., r) x (r, d)` matmul and the `(V, d)` table is never formed.
    At `V = 50k`, `d = 4096`, `r = 1` that is a 4096-fold saving per member, and the base
    `E[ids]` gather is unbatched under vmap so every member shares it — the same trick `dense`
    plays with the base GEMM.

    EGGROLL's reference implementation raises `NotImplementedError` on the embedding path
    (`docs/BACKLOG.md` B4). Nothing about the *perturbation* is hard there; what is missing is
    somewhere to intercept, because an embedding is a gather and `dense` only sees matmuls.

    A plain array takes the array branch, so `IIDGaussian` and `SeedRegenerated` are unchanged.
    """
    if isinstance(table, GatherableWeight):
        return table.gather(ids)
    if isinstance(table, StructuredWeight):
        raise TypeError(
            f"{type(table).__name__} is a StructuredWeight but not a GatherableWeight, so it "
            "can be multiplied but not indexed. Implement `gather(ids)` on it, or route this "
            "leaf through `dense` instead."
        )
    return table[ids]


def dense(x: Array, w: Array | StructuredWeight) -> Array:
    """x: (..., n), w: (m, n) or structured. Returns (..., m).

    The isinstance check is resolved at trace time, so it costs nothing at runtime.
    """
    if isinstance(w, StructuredWeight):
        return w.apply_to(x)
    return x @ w.T
