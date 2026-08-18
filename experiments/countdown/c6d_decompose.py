#!/usr/bin/env python
"""Decompose the C6d D=1 vs D=8 difference: decode chaos vs update invariance.

(a) Same seed, same members: how many members' generation-0 decoded tokens differ
    between the D=1 and D=8 compilations? (bf16 rounding differs per program;
    greedy argmax amplifies near-ties into different text.)
(b) tell with one FIXED fitness vector at D=1 and D=8: the update-path invariance
    number, reported as norm relative error per state and worst leaf allclose gap.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

HERE = Path("/root/shardes/experiments/countdown")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import task  # noqa: E402
from run_es import STRATEGIES, tokenize  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import qwen2  # noqa: E402

cfg = yaml.safe_load((HERE / "c6d-d8-g1.yaml").read_text())
ckpt = snapshot_download(cfg["model"], local_files_only=True,
                         allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(ckpt)
mcfg = qwen2.Config.qwen25_05b()
params = qwen2.load(ckpt, mcfg, dtype=jnp.bfloat16)
puzzles = task.make_puzzles(cfg["puzzle_seed"], cfg["n_puzzles"])
B = cfg["puzzles_per_gen"]
ids, plen = tokenize(tok, puzzles[:B], cfg["pad_to"])
_, all_plen = tokenize(tok, puzzles, cfg["pad_to"])
prefill_len = int(all_plen.min())


def model(p, batch):
    i, l = batch
    return qwen2.generate(p, i, l, mcfg, cfg["max_new"], prefill=prefill_len)


outs, states, perts, es_by_d = {}, {}, {}, {}
for d in (1, 8):
    mesh = sharding.make_mesh(d)
    es = ShardedES(STRATEGIES[cfg["strategy"]](cfg), n=cfg["population"],
                   sigma=cfg["sigma"], lr=cfg["lr"], mesh=mesh,
                   compute_dtype=jnp.bfloat16)
    state = es.init(jax.random.key(cfg["seed"]), params)
    pert, state = es.ask(state)
    gen = np.asarray(jax.jit(lambda s, p, i, l: es.apply(model, s, p)((i, l)))(
        state, pert, ids, plen))
    outs[d], states[d], perts[d], es_by_d[d] = gen, state, pert, es
    print(f"D={d}: evaluated {gen.shape}", flush=True)

diff = [m for m in range(cfg["population"])
        if not np.array_equal(outs[1][m], outs[8][m])]
print(f"(a) members whose generation-0 tokens differ: {len(diff)}/{cfg['population']}"
      f" {diff[:8]}")

fit = jnp.linspace(-1.0, 1.0, cfg["population"], dtype=jnp.float32)
upd = {}
for d in (1, 8):
    new = jax.jit(es_by_d[d].tell)(states[d], perts[d], fit)
    upd[d] = jax.tree.map(lambda a, b: np.asarray(a, np.float64) - np.asarray(b, np.float64),
                          new.params, states[d].params)
    print(f"D={d}: tell done", flush=True)

la, lb = jax.tree.leaves(upd[1]), jax.tree.leaves(upd[8])
num = np.sqrt(sum(float(np.sum((a - b) ** 2)) for a, b in zip(la, lb)))
den = np.sqrt(sum(float(np.sum(b ** 2)) for b in lb))
worst = max(float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-30)
            for a, b in zip(la, lb))
print(f"(b) fixed-fitness update, D=1 vs D=8: norm rel err {num / den:.2e}, "
      f"worst leaf max-abs/leaf-scale {worst:.2e}")
