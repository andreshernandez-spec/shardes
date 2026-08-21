#!/usr/bin/env python
"""E15: estimator accuracy on the real model, closing the synthetic-to-real bridge.

    python e15_accuracy.py --config e15.yaml          # A100, ~30-60 min
    python e15_accuracy.py --config e15.yaml --smoke  # random tiny model, CPU

Phase 0 measured how accurately each perturbation strategy estimates the true
gradient, on synthetic objectives; C6b carried that to the task through fitted
curves. This measures the missing link directly: on Qwen2.5-0.5B itself, the
cosine between the ES update direction and the true gradient of the same
differentiable objective.

The objective is the teacher-forced next-token loss (qwen2.nll) over a fixed
batch of Countdown prompts: the task reward is not differentiable, but the
question here is the geometry of the estimator on a real transformer's loss
surface, and the NLL is the differentiable loss this model family trains on.
The true gradient comes from jax.grad of that exact function; the ES estimate
is one production ask/evaluate/tell step's parameter delta (lr scales out of a
cosine), with the NLL itself as the fitness (tell minimizes; the sign convention
that cost this script its first smoke), so the code path being validated is
the library's own, never a reimplementation.

Per (strategy, N): `replicates` independent ES seeds against one fixed
gradient; median and range of cosines reported, one JSON per cell. The
analysis compares these against the phase-0 fitted predictions at the same
N/d_eff (analysis_c6b's machinery), which is the bridge being tested.

Fitness is evaluated in even chunks of `member_chunk` members, because the
low-rank arms vmap every member through the forward at once and the logits
alone for 240 members are ~70 GiB on this model (both OOMs of 2026-08-21 were
this cell). Chunking is the library's own regeneration contract, not a
reimplementation: perturbations re-derive from (base_key, member_ids), so each
chunk is `strategy.sample` on a slice of ids plus `strategy.apply`, exactly
what core.py's contractions already do. Per-member fitness is independent of
how members are batched, so the cosines are unchanged. Chunks must be whole
antithetic pairs (mirrored.py raises otherwise). Cells with an existing JSON
are skipped, so a rerun resumes rather than repeats.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import task  # noqa: E402
from run_es import STRATEGIES as _E13_STRATEGIES  # noqa: E402
from run_es import tokenize  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import qwen2  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402

#: The E13 arms plus the unpaired low-rank point EGGROLL's sampler cannot
#: express; its real-model accuracy number belongs next to the mirrored one.
STRATEGIES = {**_E13_STRATEGIES, "lowrank_r1": lambda cfg: LowRank(r=1)}


def flat_dot(a, b):
    parts = jax.tree.map(lambda x, y: jnp.vdot(x.astype(jnp.float32),
                                               y.astype(jnp.float32)), a, b)
    return jax.tree.reduce(lambda x, y: x + y, parts)


def cosine(a, b) -> float:
    num = flat_dot(a, b)
    return float(num / jnp.sqrt(flat_dot(a, a) * flat_dot(b, b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="random tiny model, tiny N: exercises every code path")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    if args.smoke:
        cfg.update(populations=[4, 8], replicates=2, n_prompts=2, pad_to=32,
                   member_chunk=2,  # 2 and 4 chunks: exercises the concat path
                   results_dir=cfg["results_dir"] + "-smoke")

    mcfg = qwen2.Config.qwen25_05b()
    if args.smoke:
        mcfg = qwen2.Config(vocab=256, d_model=64, n_layers=2, n_heads=4,
                            n_kv_heads=2, d_ff=128)
        params = qwen2.init(jax.random.key(0), mcfg, dtype=jnp.bfloat16)
        ids = jax.random.randint(jax.random.key(1),
                                 (cfg["n_prompts"], cfg["pad_to"]), 0, 256)
        mask = jnp.ones_like(ids)
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415
        ckpt = snapshot_download(cfg["model"])
        tok = AutoTokenizer.from_pretrained(ckpt)
        params = qwen2.load(ckpt, mcfg, dtype=jnp.bfloat16)
        puzzles = task.make_puzzles(cfg["puzzle_seed"], cfg["n_prompts"])
        ids, plen = tokenize(tok, puzzles, cfg["pad_to"])
        ids = jnp.asarray(ids)
        mask = (jnp.arange(ids.shape[1])[None, :] < jnp.asarray(plen)[:, None])
    batch = (ids, mask.astype(jnp.int32))

    # The true gradient of the exact objective the members are scored on.
    # Computed once, in f32, against the bf16 params the ES step perturbs.
    grad = jax.jit(jax.grad(lambda p: qwen2.nll(p, batch, mcfg)))(params)
    neg_grad = jax.tree.map(lambda g: -g, grad)
    jax.block_until_ready(neg_grad)
    print(f"true gradient computed; |g| leaves: {len(jax.tree.leaves(grad))}",
          flush=True)

    mesh = sharding.make_mesh(1)
    out = HERE / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    import harness  # noqa: PLC0415

    env = harness.capture_env(HERE, (cfg["results_dir"],))

    def loss32(p, b):
        return qwen2.nll(p, b, mcfg)  # already f32 by construction

    for strategy in cfg["strategies"]:
        for n in cfg["populations"]:
            outfile = out / f"s={strategy}__N={n}.json"
            if outfile.exists():
                print(f"skip {outfile.name}: already measured", flush=True)
                continue
            chunk = min(cfg.get("member_chunk", n), n)
            cosines = []
            for rep in range(cfg["replicates"]):
                es = ShardedES(STRATEGIES[strategy](cfg), n=n,
                               sigma=cfg["sigma"], lr=1.0, mesh=mesh,
                               compute_dtype=jnp.bfloat16)
                pairing = getattr(es.strategy, "pairing", 1)
                if n % chunk or chunk % pairing:
                    raise ValueError(
                        f"member_chunk={chunk} must divide N={n} and be a "
                        f"multiple of the strategy's pairing of {pairing}")
                state = es.init(jax.random.key(1000 + rep), params)
                pert, asked = es.ask(state)

                @jax.jit
                def fitness_chunk(st, base_key, ids_c):
                    # The chunk's members regenerated from their global ids,
                    # the same re-derivation tell's contraction does.
                    view = es._view(st.params)
                    sub = es.strategy.sample(base_key, view, ids_c)
                    return es.strategy.apply(loss32, view, sub, st.sigma)(batch)

                @jax.jit
                def update(st, pert, fitness):
                    # tell MINIMIZES fitness (core.py's convention; run_es
                    # negates its reward for the same reason), so the NLL
                    # goes in as-is.
                    return es.tell(st, pert, fitness)

                t0 = time.perf_counter()
                ids = jnp.arange(n, dtype=jnp.int32)
                fitness = jnp.concatenate(
                    [fitness_chunk(asked, pert.base_key, ids[i:i + chunk])
                     for i in range(0, n, chunk)])
                new = update(asked, pert, fitness)
                est = jax.tree.map(lambda a, b: a.astype(jnp.float32)
                                   - b.astype(jnp.float32),
                                   new.params, state.params)
                c = cosine(est, neg_grad)
                cosines.append(c)
                print(f"{strategy} N={n} rep={rep}: cos {c:.5f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            rec = {"config": {"strategy": strategy, "population": n,
                              "sigma": cfg["sigma"],
                              "n_prompts": cfg["n_prompts"],
                              "member_chunk": chunk,
                              "objective": "teacher-forced NLL",
                              "smoke": bool(args.smoke)},
                   "cosines": cosines,
                   "cosine_median": statistics.median(cosines),
                   "env": env}
            outfile.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
