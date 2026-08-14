"""One transformer block. The configuration the Phase 0 result is actually about.

Six square matrices (q, k, v, o, up, down), no learnable norm parameters, single head.
All matmuls go through `shardes.nn.dense`, which is the seam a structured strategy
rewrites (docs/01 C0.1).

**Three deliberate departures from a real block**, all of which belong in the limitations
section rather than in a footnote:

`d_ff = d_model`, where a real transformer uses 4x. Square everywhere means every matrix
has the same `d_eff = m + n`, so F5's x-axis is one number instead of a mixture of 1024
and 2560. The parameter-dimension story is what the experiment is about; the FLOP ratio
is not.

No learnable RMSNorm scale, so every leaf is a matrix and `LowRank` has no special case.
Mixed leaf types are real and are exactly where EGGROLL's reference implementation raises
NotImplementedError for embeddings. That is a Phase 1 problem worth meeting deliberately
rather than tripping over here.

Single head. Multi-head is a reshape; it changes no parameter count and no `d_eff`.

At m = n = 512 the six matrices give d_eff = 1,572,864 full rank and 6,144 at rank 1, a
256x gap. See src/shardes/dimensions.py.

A single transformer block is not an LLM. The N/d_eff regime transfers; the loss landscape
does not.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shardes.nn import dense
from shardes.types import Array, Key, PyTree

NAMES = ("wq", "wk", "wv", "wo", "w_up", "w_down")


class Batch(NamedTuple):
    """Fixed input and target. No data pipeline: the objective has to be deterministic
    so that replicate-to-replicate variation is the estimator and nothing else."""

    x: Array
    target: Array


def init(key: Key, d_model: int = 512) -> dict:
    """Six (d_model, d_model) matrices, scaled 1/sqrt(d) so activations stay O(1)."""
    keys = jax.random.split(key, len(NAMES))
    scale = d_model**-0.5
    return {
        name: scale * jax.random.normal(k, (d_model, d_model), dtype=jnp.float32)
        for name, k in zip(NAMES, keys)
    }


def make_batch(key: Key, d_model: int = 512, batch: int = 8, seq: int = 32) -> Batch:
    kx, kt = jax.random.split(key)
    return Batch(
        x=jax.random.normal(kx, (batch, seq, d_model), dtype=jnp.float32),
        target=jax.random.normal(kt, (batch, seq, d_model), dtype=jnp.float32),
    )


def _rms_norm(x: Array) -> Array:
    """Fixed scale. No learnable parameters, on purpose; see the module docstring."""
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)


def forward(params: PyTree, x: Array) -> Array:
    """x: (batch, seq, d) -> (batch, seq, d). Pre-norm, residual, single head.

    Every matmul against a parameter goes through `dense`. That is the only requirement a
    structured strategy places on a model, and breaking it here would silently exclude
    this block from the low-rank path.
    """
    d = x.shape[-1]

    h = _rms_norm(x)
    q, k, v = dense(h, params["wq"]), dense(h, params["wk"]), dense(h, params["wv"])
    scores = jnp.einsum("bqd,bkd->bqk", q, k) * d**-0.5
    attn = jax.nn.softmax(scores, axis=-1)
    x = x + dense(jnp.einsum("bqk,bkd->bqd", attn, v), params["wo"])

    h = _rms_norm(x)
    return x + dense(jax.nn.gelu(dense(h, params["w_up"])), params["w_down"])


def loss(params: PyTree, batch: Batch) -> Array:
    """Mean squared error against the fixed target. Scalar.

    This is `f` in `grad f`: `shardes.estimator` applies no sign flip, so the estimate is
    the gradient of this loss, not an ascent direction.

    The reduction accumulates in f32 explicitly. A no-op for the f32 sweeps, and the
    contract under a bf16 forward: `tell` refuses sub-f32 fitness because comparisons
    below f32 have already merged members that differ, and the reduction is where f32
    has to begin.
    """
    err = forward(params, batch.x) - batch.target
    return jnp.mean(jnp.square(err), dtype=jnp.float32)


def grad(params: PyTree, batch: Batch) -> PyTree:
    """Exact gradient by backprop. The oracle every Phase 0 number is measured against.

    Not an approximation and not a reference estimator with huge N: this is the true
    gradient, which is the whole reason the harness uses a differentiable model.
    """
    return jax.grad(loss)(params, batch)
