"""Qwen2.5 against the seams. The E13 model (docs/05 C6).

Every matmul against a 2-D parameter goes through `shardes.nn.dense`, the token table
goes through `shardes.nn.embed`, and 1-D leaves (RMSNorm scales, QKV biases) are read
directly, which is safe under every strategy: only 2-D leaves become structured weights,
vectors stay plain arrays under dense perturbation.

The tied embedding is the point of this port. It is one leaf read through both seams:
`embed(E, ids)` on the way in and `dense(h, E)` as the LM head on the way out, which is
exactly the tied-weights convention (`logits = h @ E.T`). Under `LowRank` both reads see
the same rank-r correction without the `(V, d)` table ever being formed, and this is the
case EGGROLL's reference implementation raises `NotImplementedError` on (C6c).

**HF weight convention is used unchanged**: projection weights are `(out, in)` and
`dense(x, w) = x @ w.T`, so tensors load from safetensors without transposition. Names
map 1:1 to `model.layers.N.*` (see `load`).

The forward is teacher-forced scoring over a full sequence. Decode with a KV cache is
the generation loop's job and lives with the Countdown task, not here: this module is
the model, and scoring is what the logit-validation tests need.

Not an approximation of Qwen2.5: RMSNorm (eps from config), RoPE at theta from config,
GQA with repeated KV heads, SwiGLU, QKV biases, tied head. `Config.qwen25_05b()` carries
the published 0.5B hyperparameters; correctness against the released checkpoint is
`tests/test_qwen2.py`'s golden-logit tier, which skips when the checkpoint is not cached.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shardes.nn import dense, embed
from shardes.types import Array, Key, PyTree


class Config(NamedTuple):
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    rope_theta: float = 1e6
    rms_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def qwen25_05b(cls) -> "Config":
        """Qwen/Qwen2.5-0.5B(-Instruct) config.json, the E13 checkpoint."""
        return cls(vocab=151936, d_model=896, n_layers=24, n_heads=14, n_kv_heads=2,
                   d_ff=4864, rope_theta=1e6, rms_eps=1e-6)

    @classmethod
    def tiny(cls) -> "Config":
        """For CPU tests: same wiring, 300k parameters."""
        return cls(vocab=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                   d_ff=64, rope_theta=1e4)


def init(key: Key, cfg: Config, dtype=jnp.float32) -> dict:
    """Random init with the real tree structure. For tests; real weights come from `load`.

    The tree layout is the contract the tests pin: `embedding` at top level (tied, so
    there is no separate lm_head leaf), `layers` as a list of per-layer dicts.
    """
    k_embed, k_layers = jax.random.split(key)

    def linear(k, out_d, in_d):
        return (jax.random.normal(k, (out_d, in_d), jnp.float32) * in_d**-0.5).astype(dtype)

    def layer(k):
        ks = jax.random.split(k, 8)
        h = cfg.head_dim
        return {
            "ln1": jnp.ones((cfg.d_model,), dtype),
            "wq": linear(ks[0], cfg.n_heads * h, cfg.d_model),
            "bq": jnp.zeros((cfg.n_heads * h,), dtype),
            "wk": linear(ks[1], cfg.n_kv_heads * h, cfg.d_model),
            "bk": jnp.zeros((cfg.n_kv_heads * h,), dtype),
            "wv": linear(ks[2], cfg.n_kv_heads * h, cfg.d_model),
            "bv": jnp.zeros((cfg.n_kv_heads * h,), dtype),
            "wo": linear(ks[3], cfg.d_model, cfg.n_heads * h),
            "ln2": jnp.ones((cfg.d_model,), dtype),
            "w_gate": linear(ks[4], cfg.d_ff, cfg.d_model),
            "w_up": linear(ks[5], cfg.d_ff, cfg.d_model),
            "w_down": linear(ks[6], cfg.d_model, cfg.d_ff),
        }

    return {
        "embedding": (jax.random.normal(k_embed, (cfg.vocab, cfg.d_model), jnp.float32)
                      * 0.02).astype(dtype),
        "layers": [layer(k) for k in jax.random.split(k_layers, cfg.n_layers)],
        "ln_f": jnp.ones((cfg.d_model,), dtype),
    }


def _rms_norm(x: Array, scale: Array, eps: float) -> Array:
    # Variance in f32 whatever the compute dtype: a mean of bf16 squares loses the very
    # distinctions RMSNorm exists to preserve, and every mixed-precision LM does this.
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x.astype(jnp.float32) * jax.lax.rsqrt(var + eps)).astype(x.dtype) * scale


def _rope(x: Array, theta: float) -> Array:
    """x: (..., seq, heads, head_dim). Rotate-half convention, angles in f32."""
    seq, hd = x.shape[-3], x.shape[-1]
    pos = jnp.arange(seq, dtype=jnp.float32)
    inv = theta ** (-jnp.arange(0, hd, 2, dtype=jnp.float32) / hd)
    ang = pos[:, None] * inv[None, :]                      # (seq, hd/2)
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    cos = jnp.concatenate([cos, cos], axis=-1)[:, None, :]  # (seq, 1, hd)
    sin = jnp.concatenate([sin, sin], axis=-1)[:, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (x.astype(jnp.float32) * cos + rotated.astype(jnp.float32) * sin).astype(x.dtype)


def _attention(p: dict, x: Array, cfg: Config) -> Array:
    """x: (batch, seq, d). Causal, GQA via repeated KV heads."""
    b, s, _ = x.shape
    hd, nh, nkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads

    q = (dense(x, p["wq"]) + p["bq"]).reshape(b, s, nh, hd)
    k = (dense(x, p["wk"]) + p["bk"]).reshape(b, s, nkv, hd)
    v = (dense(x, p["wv"]) + p["bv"]).reshape(b, s, nkv, hd)
    q, k = _rope(q, cfg.rope_theta), _rope(k, cfg.rope_theta)
    k = jnp.repeat(k, nh // nkv, axis=2)
    v = jnp.repeat(v, nh // nkv, axis=2)

    # Scores in f32: a bf16 softmax over 32k positions is its own experiment.
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * hd**-0.5
    scores = jnp.where(jnp.tril(jnp.ones((s, s), bool)), scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, nh * hd)
    return dense(out, p["wo"])


def forward(params: PyTree, ids: Array, cfg: Config) -> Array:
    """ids: (batch, seq) int32 -> logits (batch, seq, vocab), teacher-forced.

    Logits in f32: they feed a loss or a softmax, and both are reductions.
    """
    x = embed(params["embedding"], ids)
    for p in params["layers"]:
        h = _rms_norm(x, p["ln1"], cfg.rms_eps)
        x = x + _attention(p, h, cfg)
        h = _rms_norm(x, p["ln2"], cfg.rms_eps)
        x = x + dense(jax.nn.silu(dense(h, p["w_gate"])) * dense(h, p["w_up"]), p["w_down"])
    x = _rms_norm(x, params["ln_f"], cfg.rms_eps)
    return dense(x, params["embedding"]).astype(jnp.float32)  # tied head: h @ E.T


def nll(params: PyTree, batch, cfg: Config) -> Array:
    """Mean next-token NLL over the batch, masked. f32 by construction (logits are).

    `batch` is `(ids, mask)`: predict `ids[:, 1:]` from `ids[:, :-1]`, count positions
    where `mask[:, 1:]` is set. The E13 fitness for scoring-style probes and the loss
    the golden tests compare; the generation-based Countdown reward lives with the task.
    """
    ids, mask = batch
    logits = forward(params, ids[:, :-1], cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    tok = jnp.take_along_axis(logp, ids[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(jnp.float32)
    return -jnp.sum(tok * m) / jnp.maximum(jnp.sum(m), 1.0)


# ------------------------------------------------------------------------------------


HF_LAYER = {
    "ln1": "input_layernorm.weight",
    "wq": "self_attn.q_proj.weight", "bq": "self_attn.q_proj.bias",
    "wk": "self_attn.k_proj.weight", "bk": "self_attn.k_proj.bias",
    "wv": "self_attn.v_proj.weight", "bv": "self_attn.v_proj.bias",
    "wo": "self_attn.o_proj.weight",
    "ln2": "post_attention_layernorm.weight",
    "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight",
    "w_down": "mlp.down_proj.weight",
}


def load(checkpoint_dir, cfg: Config, dtype=jnp.bfloat16) -> dict:
    """Params from a HF safetensors checkpoint directory. `(out, in)` kept as stored.

    safetensors is imported here rather than at module top so the library keeps its
    dependency floor: importing `shardes.problems.qwen2` needs jax, loading a real
    checkpoint needs the file that only exists if the user fetched one.
    """
    from pathlib import Path  # noqa: PLC0415

    files = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {checkpoint_dir}")

    # Import after the directory check: the missing-checkpoint error must be reachable
    # without the optional dependency, or the wiring tests need safetensors to fail well.
    from safetensors import safe_open  # noqa: PLC0415
    raw = {}
    for f in files:
        with safe_open(f, framework="numpy") as sf:
            for name in sf.keys():
                raw[name] = sf.get_tensor(name)

    def take(name):
        # bf16 checkpoints arrive as uint16 views under numpy; jnp.asarray + view
        # handles both that and ordinary f32 tensors.
        t = raw.pop(name)
        arr = jnp.asarray(t)
        if t.dtype.itemsize == 2 and t.dtype.kind == "u":
            arr = arr.view(jnp.bfloat16)
        return arr.astype(dtype)

    params = {
        "embedding": take("model.embed_tokens.weight"),
        "ln_f": take("model.norm.weight"),
        "layers": [
            {ours: take(f"model.layers.{i}.{theirs}") for ours, theirs in HF_LAYER.items()}
            for i in range(cfg.n_layers)
        ],
    }
    raw.pop("lm_head.weight", None)  # present in some exports; tied here, so unused
    if raw:
        raise ValueError(f"unmapped tensors in checkpoint: {sorted(raw)[:5]}")
    return params
