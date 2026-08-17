"""LowRank(r): EGGROLL. Perturbation E = (1/sqrt(r)) A B^T, never materialized.

    A: (n_members, m, r)
    B: (n_members, k, r)     for a leaf of shape (m, k)

`apply` substitutes a `LowRankWeight` into the params tree and the model's `nn.dense`
dispatches to it, rewriting `x @ W.T` into `x @ W.T + (x @ B) @ A.T`. The base term has no
batched operand, so vmap leaves it unbatched and every member shares one GEMM. That is
EGGROLL's whole trick and here it falls out of vmap rather than being arranged by hand.

`contract` is `sum_n w_n a_n b_n^T`, one (m x nr) by (nr x k) GEMM: m*k*n*r FLOPs. At
m = k = 4096, n = 2^18, r = 1 that is 4.4 TFLOP, about 6 ms on an H100. The cost that
bites is storage: A and B together are n*r*(m+k), which at those numbers is 4.3 GB per
layer in bf16, or half that under `Mirrored` because an antithetic pair shares its factors.
That is what motivates regenerating them from seeds during contraction
(docs/00-context.md).

**Leaves that are not rank-2 are perturbed densely.** A vector has no low-rank
factorisation, and `E = a b^T` for a (d,) leaf is just a scaled `a`. This matters more
than it looks: it is the same case EGGROLL's reference implementation raises
NotImplementedError for on embeddings, and it means `LowRank` on the quadratic problem
degenerates exactly to `IIDGaussian`, which is a useful cross-check rather than a bug.

Invariant 3 (CLAUDE.md): an (n_members, m, k) array under this path is a bug. There is a
jaxpr test.
"""

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from shardes.coupling import GAUSSIAN, Coupling
from shardes.strategies._noise import leaf_streams
from shardes.strategies._scale import Separable, densify, is_separable, per_leaf
from shardes.strategies._select import mapped, rows, rows_like, unmapped
from shardes.types import Array, Key, PyTree


def _not_an_array(op: str, seam: str):
    raise TypeError(
        f"LowRankWeight does not support {op}. It is a structured weight standing in for "
        f"`W + scale * A B^T`, and the whole point is that the sum is never formed, so it "
        f"cannot be used as an array. Route this leaf through `shardes.nn.{seam}` instead. "
        "docs/01-phase0-estimator-harness.md C0.1."
    )


@register_dataclass
@dataclass(frozen=True)
class LowRankWeight:
    """W + scale * A B^T, applied without ever forming the sum.

    Satisfies `shardes.nn.StructuredWeight` and `GatherableWeight` structurally, so `dense`
    and `embed` dispatch to it.

    **A registered dataclass rather than a NamedTuple, and that is a correctness fix.** As a
    NamedTuple it inherited tuple semantics, so a model that wrote the natural thing got a
    wrong answer *silently* rather than an error:

        table[0]         -> the base matrix `w`, not row 0
        len(table)       -> 4, the field count, not the vocabulary size
        for row in table -> iterates the four fields

    None of those raised. A dataclass is not a sequence, so all three are now loud, and the
    dunders below turn the remaining array-shaped mistakes into a message that names the seam
    to use. Being a pytree is unaffected: four leaves either way, and it travels through jit,
    vmap and shard_map the same.
    """

    w: Array  # (m, k) base, unbatched under vmap so its GEMM is shared
    a: Array  # (m, r)
    b: Array  # (k, r)
    scale: Array  # sigma / sqrt(r)

    #: Metadata is safe to expose and useful for a caller's own shape assertions. Data is not.
    @property
    def shape(self):
        return self.w.shape

    @property
    def dtype(self):
        return self.w.dtype

    @property
    def T(self):
        _not_an_array("transposition", "dense")

    def __getitem__(self, _i):
        _not_an_array("indexing", "embed")

    def __iter__(self):
        _not_an_array("iteration", "embed")

    def __len__(self):
        _not_an_array("len()", "embed")

    def __array__(self, *_a, **_k):
        _not_an_array("conversion to a numpy array", "dense")

    def __mul__(self, _o):
        _not_an_array("elementwise arithmetic", "dense")

    __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __mul__

    def __matmul__(self, _o):
        _not_an_array("`@`", "dense")

    __rmatmul__ = __matmul__

    def apply_to(self, x: Array) -> Array:
        return x @ self.w.T + ((x @ self.b) @ self.a.T) * self.scale

    def gather(self, ids: Array) -> Array:
        """`(W + scale * A B^T)[ids]`, without forming the sum. See `shardes.nn.embed`.

        Exact rather than approximate: indexing distributes over the sum, so this is the
        same identity `apply_to` uses, read the other way round. `self.w[ids]` is unbatched
        under vmap, so members share the base gather.
        """
        return self.w[ids] + (self.a[ids] @ self.b.T) * self.scale


#: Raised when a per-coordinate sigma meets a structured leaf. The message explains the
#: mathematics rather than the shapes, because the shapes are a symptom.
#:
#: The perturbed weight is `W + sigma * (A B^T)` with `*` elementwise. An elementwise sigma
#: does not preserve the rank of that product: measured at `m=5, k=4, r=2`, the rank of
#: `sigma * (A B^T)` is 4. So there is no pair of GEMMs left to apply, and materializing the
#: sum would discard invariant 3. `core.State` documents sigma as "a scalar or a
#: params-shaped pytree", which is true of the full-rank strategies and was never true here.
#:
#: The separable family survives and is the natural extension if anyone needs it:
#: `(u v^T) * (A B^T) == (u[:, None] * A) @ (v[:, None] * B).T`, still rank `r`, still two
#: GEMMs. That covers per-output-unit and per-input-unit step sizes, which is what a
#: per-coordinate schedule usually means in practice. It is not implemented: it would make a
#: sigma leaf a pair rather than an array, which is a change to the public contract.
_NON_SCALAR_SIGMA = (
    "LowRank got a sigma of shape {shape} for a leaf it perturbs with rank-r factors, and "
    "only a scalar works there. The perturbation is W + sigma * (A B^T), and an elementwise "
    "sigma raises the rank of that product above r, so it cannot be applied as two GEMMs "
    "against the factors and the sum is never formed (invariant 3). Use a scalar sigma for "
    "this leaf, or a full-rank strategy if the per-coordinate schedule is the point. "
    "docs/proposal-review-fixes.md has the reasoning and the separable case that would work."
)


_SEPARABLE_ON_DENSE = (
    "a Separable sigma was given for a leaf of shape {shape}, which LowRank perturbs densely "
    "rather than with factors. Separable scales an output axis and an input axis, and a leaf "
    "that is not rank 2 has only one. Use a scalar or an ordinary per-coordinate array there."
)


def _check_separable(s, leaf) -> None:
    """Shapes, at trace time, where the message can still name the leaf.

    Without this a mismatched `u` broadcasts against `a` and fails several lines later inside
    a GEMM, reported against shapes the caller never wrote.
    """
    m, k = leaf.shape
    if s.u.shape != (m,) or s.v.shape != (k,):
        raise ValueError(
            f"Separable sigma for a {leaf.shape} leaf needs u of shape ({m},) and v of shape "
            f"({k},), got u{s.u.shape} and v{s.v.shape}. u scales the output axis and v the "
            "input axis, matching `W + (u v^T) * (A B^T)`."
        )


class LeafFactors(NamedTuple):
    """Per-leaf perturbation state. `b is None` marks a densely perturbed leaf.

    None is an empty pytree node, so a dense leaf simply contributes no `b` array rather
    than needing a sentinel or a parallel structure.
    """

    a: Array
    b: Array | None


def _is_factors(node) -> bool:
    return isinstance(node, LeafFactors)


class LowRankPerturbation(NamedTuple):
    base_key: Key
    member_ids: Array
    factors: PyTree  # params-shaped, a LeafFactors at each leaf position


class LowRank:
    """Rank-r factored perturbation. r = 1 is enough in practice (Sarkar et al.).

    `coupling` designs the a- and b-vectors across members. This is the panel the whole
    coupling question is about: sampling happens in R^(m+r) and R^(k+r) rather than R^(mk),
    so N/d_eff crosses 1 at populations that fit on a GPU (docs/00-context.md).

    Each of the r columns is its own design family, so column j of every member's A is
    coupled against column j of every other member's, and never against column j+1. Mixing
    them would couple directions that are then summed, which is not the same point set.
    """

    def __init__(self, r: int = 1, coupling: Coupling = GAUSSIAN):
        if r < 1:
            raise ValueError(f"rank must be at least 1, got {r}")
        self.r = int(r)
        self.coupling = coupling

    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> LowRankPerturbation:
        """Unit-variance E per entry: E[E_ij^2] = (1/r) sum_k E[a_ik^2] E[b_jk^2] = 1."""
        leaves, treedef = jax.tree.flatten(params)
        streams = leaf_streams(base_key, len(leaves))
        # Split off the design families outside the vmap: they are per (leaf, side, column)
        # and must not depend on the member.
        columns = [
            None if x.ndim != 2 else jax.random.split(s, 2 * self.r)
            for s, x in zip(streams, leaves)
        ]

        def one(i: Array) -> PyTree:
            out = []
            for leaf, stream, cols in zip(leaves, streams, columns):
                if cols is None:
                    a = self.coupling(stream, i, leaf.size, leaf.dtype).reshape(leaf.shape)
                    out.append(LeafFactors(a, None))
                    continue
                m, k = leaf.shape
                # vmap over the column keys, not a Python loop: the loop unrolled 2r
                # coupling subgraphs per leaf into the trace, and sample is regenerated
                # inside tell's jitted graph, so XLA compile time scaled with r
                # (measured 63 s at r=1 vs 963 s at r=16 on CPU at 12 layers; 6600 s
                # on an A100 at 0.5B). vmap of a draw over keys is defined to equal
                # the stacked per-key calls, so the noise is bit-identical; the seed
                # contract test holds it there.
                stack = lambda d, ks: jax.vmap(  # noqa: E731
                    lambda c: self.coupling(c, i, d, leaf.dtype), out_axes=-1
                )(ks)
                out.append(LeafFactors(stack(m, cols[: self.r]), stack(k, cols[self.r :])))
            return jax.tree.unflatten(treedef, out)

        return LowRankPerturbation(base_key, member_ids, jax.vmap(one)(member_ids))

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        params: PyTree,
        pert: LowRankPerturbation,
        sigma: float,
    ) -> Callable[[Array], Array]:
        """The perturbed weights are never formed. See the module docstring."""
        # Per leaf, so a params-shaped diagonal works exactly as a scalar does. The
        # 1/sqrt(r) is folded in here rather than at sample time because it belongs to the
        # scale, not to the factors (docs/02 C1.4).
        sigmas = per_leaf(sigma, params)

        def g(x: Array) -> Array:
            def one(factors: PyTree) -> Array:
                def substitute(leaf, s, lf):
                    if lf.b is None:
                        # A densely perturbed leaf never enters the two-GEMM identity, so
                        # any shape of sigma multiplies it the ordinary way. A Separable has
                        # no meaning here: there is no second axis to scale.
                        if is_separable(s):
                            raise ValueError(_SEPARABLE_ON_DENSE.format(shape=leaf.shape))
                        # In the leaf's dtype, like the factored paths below: an f32 sigma
                        # must not promote a bf16 forward back to f32.
                        return leaf + jnp.asarray(s, leaf.dtype) * lf.a
                    if is_separable(s):
                        # `(u v^T) * (A B^T) == (u * A) @ (v * B).T`. Folded into the factors,
                        # so the (m, k) product is still never formed and the forward pass is
                        # still two GEMMs. This is the whole reason the type exists.
                        _check_separable(s, leaf)
                        return LowRankWeight(
                            leaf,
                            (s.u[:, None] * lf.a).astype(leaf.dtype),
                            (s.v[:, None] * lf.b).astype(leaf.dtype),
                            jnp.asarray(1.0 / math.sqrt(self.r), leaf.dtype),
                        )
                    if jnp.ndim(s):
                        raise ValueError(_NON_SCALAR_SIGMA.format(shape=jnp.shape(s)))
                    return LowRankWeight(leaf, lf.a, lf.b,
                                         jnp.asarray(s / math.sqrt(self.r), leaf.dtype))

                return model(jax.tree.map(substitute, params, sigmas, factors), x)

            return jax.vmap(one)(pert.factors)

        return g

    def split(self, pert: LowRankPerturbation, n_rows: int):
        """`a` and `b` are `(n, m, r)` and `(n, k, r)`, so the member axis leads both.

        `b is None` marks a densely perturbed leaf, and `None` is an empty pytree node, so
        `tree.map` skips it in both the reshape and the `in_axes` without a branch here.
        """
        return (
            LowRankPerturbation(pert.base_key, rows(pert.member_ids, n_rows),
                                rows_like(pert.factors, n_rows)),
            LowRankPerturbation(None, 0, mapped(pert.factors)),
        )

    def contract(self, pert: LowRankPerturbation, weights: Array) -> PyTree:
        """sum_n w_n E_n, params-shaped, unit scale.

        The only place a full (m, k) tensor is instantiated, and it is the *update*, not
        any member's perturbation.
        """
        w = weights.astype(jnp.float32)
        scale = 1.0 / math.sqrt(self.r)

        def leaf(lf: LeafFactors):
            a = lf.a.astype(jnp.float32)
            if lf.b is None:
                return jnp.einsum("n,n...->...", w, a)
            return scale * jnp.einsum("n,nmr,nkr->mk", w, a, lf.b.astype(jnp.float32))

        return jax.tree.map(leaf, pert.factors, is_leaf=_is_factors)
