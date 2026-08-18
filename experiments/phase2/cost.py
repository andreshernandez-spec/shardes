#!/usr/bin/env python
"""E8: the low-rank vs dense cost surface, one device, one platform per run.

    python cost.py --config cost-sweep.yaml --dry-run
    python cost.py --config cost-sweep.yaml

C4's question is where EGGROLL's rewrite pays and whether the answer is
platform-shaped: naive dense ES is memory-bound on GPU because N distinct weight
matrices share no GEMM, while a TPU's MXU makes dense matmuls comparatively cheap
and 16 GB/chip makes capacity the binding constraint. This sweep measures
per-generation wall time and peak device memory over
(d_model, N, strategy, compute dtype) at D=1, on whatever device is visible; the
surface is the deliverable and the cross-platform comparison happens in the plot,
not here.

Same guards as run.py, same reasons: warmup discarded, block_until_ready on every
timed repeat, median with IQR, the trajectory guard at matmul precision `highest`
while the timed generations run at the config's precision, both recorded. One
JSON per configuration, resumable, and a configuration whose population does not
fit its device is recorded as undersized and skipped, which is a result about
memory rather than a failure (that ceiling is part of what E8 exists to map).

D=1 on purpose. The arithmetic-intensity story is per-device; scaling and
communication live in run.py's sweep, and mixing them would make both unreadable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.iid_gaussian import IIDGaussian  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

OUTPUTS = ("results-cost",)

STRATEGIES = {
    "iid_gaussian": IIDGaussian,
    "seed_regenerated": SeedRegenerated,
    "mirrored_seed": lambda: Mirrored(SeedRegenerated()),
    "mirrored_lr1": lambda: Mirrored(LowRank(r=1)),
    "mirrored_lr4": lambda: Mirrored(LowRank(r=4)),
    "mirrored_lr16": lambda: Mirrored(LowRank(r=16)),
}

SEED, SIGMA, LR = 0, 0.01, 0.05


@dataclass(frozen=True)
class Config:
    d_model: int
    population: int
    strategy: str
    dtype: str  # "float32" | "bfloat16", the compute dtype through the policy-A view

    def slug(self) -> str:
        return (f"d={self.d_model}__N={self.population}"
                f"__s={self.strategy}__dtype={self.dtype}")


def expand(cfg: dict) -> list[Config]:
    return [
        Config(d, n, s, t)
        for d in cfg["d_model"]
        for n in cfg["population"]
        for s in cfg["strategy"]
        for t in cfg["dtype"]
    ]


def measure(config: Config, cfg: dict, results: Path) -> dict:
    warmup = int(cfg.get("warmup", 3))
    repeats = int(cfg.get("repeats", 5))
    precision = cfg.get("matmul_precision", "default")

    mesh = sharding.make_mesh(1)
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=config.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=config.d_model,
        batch=int(cfg.get("batch", 8)), seq=int(cfg.get("seq", 32)),
    )
    compute = None if config.dtype == "float32" else jnp.bfloat16
    es = ShardedES(STRATEGIES[config.strategy](), n=config.population, sigma=SIGMA,
                   lr=LR, mesh=mesh, compute_dtype=compute)
    state = es.init(key, params)

    def loss32(p, b):
        # policy A: the forward may be bf16, the loss reduction is where f32 begins.
        return transformer_block.loss(p, b).astype(jnp.float32)

    @jax.jit
    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(loss32, state, pert)(batch)
        return es.tell(state, pert, fitness)

    with jax.default_matmul_precision(precision):
        for _ in range(warmup):
            state = generation(state)
        jax.block_until_ready(state.params)
        seconds = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            state = generation(state)
            jax.block_until_ready(state.params)
            seconds.append(time.perf_counter() - t0)

    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "config": config.__dict__,
        "seconds_all": seconds,
        "seconds_median": statistics.median(seconds),
        "seconds_iqr": [sorted(seconds)[len(seconds) // 4],
                        sorted(seconds)[-1 - len(seconds) // 4]],
        "matmul_precision": precision,
        "peak_bytes_device": stats.get("peak_bytes_in_use"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even when undersized configurations were skipped")
    args = ap.parse_args(argv)
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(args.config.read_text())
    results = HERE / cfg.get("results_dir", "results-cost")
    configs = expand(cfg)
    pending = [c for c in configs if not (results / f"{c.slug()}.json").exists()]
    dev = jax.local_devices()[0]
    print(f"{len(configs)} configs, {len(pending)} pending, device {dev.device_kind}")
    if args.dry_run:
        for c in pending:
            print(" ", c.slug())
        return 0

    results.mkdir(exist_ok=True)
    env = harness.capture_env(HERE, (*OUTPUTS, cfg.get("results_dir", "results-cost")))
    skipped = []
    for i, c in enumerate(pending):
        print(f"[{i + 1}/{len(pending)}] {c.slug()}", flush=True)
        try:
            rec = measure(c, cfg, results)
        except (jax.errors.JaxRuntimeError, RuntimeError) as e:
            if "RESOURCE_EXHAUSTED" not in str(e) and "Out of memory" not in str(e):
                raise
            rec = {"config": c.__dict__, "undersized": True,
                   "device": dev.device_kind}
            skipped.append(c.slug())
            print(f"  undersized on {dev.device_kind}: recorded and skipped", flush=True)
        rec["env"] = env
        (results / f"{c.slug()}.json").write_text(json.dumps(rec, indent=1))

    print(f"done: {len(pending) - len(skipped)} measured, {len(skipped)} undersized")
    if skipped and not args.allow_partial:
        print("undersized configurations present; pass --allow-partial to accept a "
              "partial surface (the skips are recorded either way)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
