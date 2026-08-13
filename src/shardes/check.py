"""`check_model`: does this model reach its weights through the seams?

    from shardes.check import check_model
    print(check_model(my_model, params, batch))     # [] means it is fine

A structured strategy (`LowRank`) substitutes a `LowRankWeight` into the params tree, so a
model must route matmuls through `shardes.nn.dense` and gathers through `shardes.nn.embed`.
Anything else raises. That is safe — it never computes the wrong thing — but the failure
arrives when the strategy runs, which may be several minutes into a sweep on rented hardware.

This gets the same answer in a second, on CPU, at whatever shapes the model already has, and
reports **every** offending leaf rather than the first.

**Why a probe rather than a jaxpr pass.** Reading the jaxpr and looking for `dot_general`
operations that touch a parameter cannot tell a matmul that went through `dense` from one that
did not: both lower to the same primitive. The seams are the only place the distinction
exists, so the check has to be made there. A probe substituted for one leaf at a time records
which seam was used and refuses everything else, which is exactly the information wanted and
needs no interpreter.

One leaf at a time, so a model with three unrouted weights produces three findings instead of
failing on the first and hiding the rest.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from shardes.types import Array, PyTree


class _Probe:
    """Stands in for one parameter leaf and records how the model reached it.

    Deliberately not a pytree: `check_model` calls the model directly rather than through a
    strategy's `apply`, so the probe never has to survive `jit` or `vmap`. Keeping it a plain
    object means the error messages can carry the leaf's path, which is the whole value.
    """

    def __init__(self, w: Array, path: str):
        self.w = w
        self.path = path
        self.seams: set[str] = set()

    # the two legal routes
    def apply_to(self, x: Array) -> Array:
        self.seams.add("dense")
        return x @ self.w.T

    def gather(self, ids: Array) -> Array:
        self.seams.add("embed")
        return self.w[ids]

    # metadata is fine; data is not
    @property
    def shape(self):
        return self.w.shape

    @property
    def dtype(self):
        return self.w.dtype

    def _refuse(self, op: str) -> Any:
        raise TypeError(
            f"{self.path} is reached by {op}, which a structured weight cannot support. "
            f"Route it through shardes.nn.dense (for `x @ W.T`) or shardes.nn.embed "
            f"(for `W[ids]`)."
        )

    @property
    def T(self):
        self._refuse("transposition, so probably `x @ W.T`")

    def __getitem__(self, _i):
        self._refuse("indexing")

    def __iter__(self):
        self._refuse("iteration")

    def __len__(self):
        self._refuse("len()")

    def __array__(self, *_a, **_k):
        self._refuse("conversion to an array")

    def __mul__(self, _o):
        self._refuse("elementwise arithmetic")

    __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __mul__

    def __matmul__(self, _o):
        self._refuse("`@`")

    __rmatmul__ = __matmul__


def _opaque(path: str, exc: Exception) -> str:
    """Translate JAX's own rejection of the probe into a finding that names the cause.

    The probe refuses the array operations it can intercept and says which seam to use. It
    cannot intercept everything: `jnp.square(leaf)` reaches JAX's abstractification first and
    comes back as "is not a valid JAX type", naming the probe's repr. `__jax_array__` looks
    like the hook for this and is not one, JAX 0.11 raises rather than calling it, which turns
    a clear TypeError into a ValueError and loses the finding entirely. So the translation
    happens here, where the path is still in scope.

    **The pytree case is the one worth spelling out**, because it is the only one that is
    silently wrong at runtime rather than loud. Everything else the probe catches would also
    raise under a real `LowRankWeight`.
    """
    return (
        f"{path} is reached as a bare array, which a structured weight cannot support. "
        f"Route it through shardes.nn.dense (for `x @ W.T`) or shardes.nn.embed (for "
        f"`W[ids]`). JAX reported: {type(exc).__name__}: {str(exc)[:120]}\n"
        f"    If this came from jax.tree.leaves, jax.tree.map or anything else that walks "
        f"the params tree, read on. Under LowRank this leaf is a LowRankWeight, a registered "
        f"pytree of (w, a, b, scale), so a tree walk descends into the factors and returns "
        f"four ordinary arrays instead of the weight they stand for. At runtime that does "
        f"NOT raise and does not give a wrong shape: it computes a different function, "
        f"quietly. A tree-shaped objective (weight decay, a parameter norm, a regulariser) "
        f"has to be written against the seams. docs/01-phase0-estimator-harness.md C0.1."
    )


def _paths(params: PyTree):
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        yield jax.tree_util.keystr(path), leaf


def check_model(
    model: Callable[..., Array], params: PyTree, *args: Any, structured_only: bool = True
) -> list[str]:
    """Findings, one per leaf the model does not reach through a seam. `[]` means fine.

    `structured_only` skips leaves a structured strategy would perturb densely anyway — those
    stay ordinary arrays under `LowRank`, so how the model touches them is its own business.
    Set it False to audit every leaf, which is what a *future* structured scheme covering
    vectors would need.

    Unused leaves are reported too, with a different message. A parameter no forward pass
    touches is still perturbed and still contracted, so it costs population variance for
    nothing — a quieter bug than an unrouted matmul and a more embarrassing one.
    """
    findings = []
    for path, leaf in _paths(params):
        if structured_only and getattr(leaf, "ndim", 0) != 2:
            continue
        probe = _Probe(leaf, path)
        probed = _substitute(params, path, probe)
        try:
            model(probed, *args)
        except (TypeError, ValueError) as exc:
            findings.append(str(exc) if path in str(exc) else _opaque(path, exc))
            continue
        if not probe.seams:
            findings.append(
                f"{path} is never read by the model. It is still perturbed and still "
                "contracted, so it spends population variance on a parameter that cannot "
                "affect the fitness."
            )
    return findings


def _substitute(params: PyTree, target: str, probe: _Probe) -> PyTree:
    """`params` with the leaf at `target` replaced by `probe`."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    return jax.tree_util.tree_unflatten(
        treedef,
        [probe if jax.tree_util.keystr(p) == target else v for p, v in leaves],
    )
