#!/usr/bin/env python
"""E17: the contraction crossover measured on the real model.

    python e17_systems.py --config e17.yaml           # 8 real devices
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      python e17_systems.py --config e17.yaml --smoke # tiny model, CPU

Times one complete production update (ask, evaluate the teacher-forced NLL,
tell) on Qwen2.5-0.5B, per (strategy, contraction, N, D) cell, with the
scaling sweeps' discipline: warm-up generations discarded (the first
compiles), timed repeats fenced with block_until_ready, median and spread
recorded, matmul precision pinned and stamped. The predictions this confronts
are frozen in e17.yaml's header; the model and prompt batch are E15's.

Cells that exceed a device's memory are recorded as OOM results rather than
resized, the matched-shapes rule. One JSON per cell, written as soon as the
cell finishes, so a killed session resumes by skipping existing files.
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
from run_es import STRATEGIES  # noqa: E402
from run_es import tokenize  # noqa: E402

import harness  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import qwen2  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="random tiny model on simulated devices")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    if args.smoke:
        cfg.update(populations=[16, 32], n_prompts=2, pad_to=32,
                   warmup=1, repeats=2,
                   results_dir=cfg["results_dir"] + "-smoke")

    jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])

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

    out = HERE / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    env = harness.capture_env(HERE, (cfg["results_dir"],))

    def loss32(p, b):
        return qwen2.nll(p, b, mcfg)

    devices = [d for d in cfg["devices"] if d <= len(jax.devices())]
    if devices != cfg["devices"]:
        print(f"only {len(jax.devices())} devices; running D={devices}", flush=True)

    for d_count in devices:
        mesh = sharding.make_mesh(d_count)
        for strategy in cfg["strategies"]:
            for how in cfg["hows"]:
                for n in cfg["populations"]:
                    name = f"s={strategy}__how={how}__N={n}__D={d_count}.json"
                    outfile = out / name
                    if outfile.exists():
                        print(f"skip {name}: already measured", flush=True)
                        continue
                    rec = {"config": {"strategy": strategy, "how": how,
                                      "population": n, "devices": d_count,
                                      "sigma": cfg["sigma"],
                                      "n_prompts": cfg["n_prompts"],
                                      "matmul_precision": cfg["matmul_precision"],
                                      "smoke": bool(args.smoke)},
                           "env": env}
                    try:
                        es = ShardedES(STRATEGIES[strategy](cfg), n=n,
                                       sigma=cfg["sigma"], lr=1.0, mesh=mesh,
                                       how=how, compute_dtype=jnp.bfloat16)
                        state = es.init(jax.random.key(7), params)

                        @jax.jit
                        def generation(state):
                            pert, state = es.ask(state)
                            fitness = es.apply(loss32, state, pert)(batch)
                            return es.tell(state, pert, fitness)

                        for _ in range(cfg["warmup"]):
                            state = generation(state)
                        jax.block_until_ready(state.params)
                        seconds = []
                        for _ in range(cfg["repeats"]):
                            t0 = time.perf_counter()
                            state = generation(state)
                            jax.block_until_ready(state.params)
                            seconds.append(time.perf_counter() - t0)
                        rec["seconds_all"] = seconds
                        rec["seconds_median"] = statistics.median(seconds)
                        print(f"{strategy} how={how} N={n} D={d_count}: "
                              f"{rec['seconds_median'] * 1e3:.1f} ms", flush=True)
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).splitlines()[0] if str(e) else repr(e)
                        if "RESOURCE_EXHAUSTED" not in msg:
                            raise
                        rec["status"] = "oom"
                        rec["error"] = msg[:500]
                        print(f"{strategy} how={how} N={n} D={d_count}: OOM "
                              "(recorded)", flush=True)
                    outfile.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
