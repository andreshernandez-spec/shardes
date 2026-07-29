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


def dense(x: Array, w: Array | StructuredWeight) -> Array:
    """x: (..., n), w: (m, n) or structured. Returns (..., m).

    The isinstance check is resolved at trace time, so it costs nothing at runtime.
    """
    if isinstance(w, StructuredWeight):
        return w.apply_to(x)
    return x @ w.T
