#!/usr/bin/env python
"""E10: what the shaping barrier costs, measured directly.

    python barrier.py --config barrier-tpu.yaml --dry-run
    XLA_FLAGS=--xla_force_host_platform_device_count=8 python barrier.py --config ...

`comms.py` measured the barrier by compiling against `shaping=none` and
subtracting, and that difference now reads 0 for the wrong reason: `tell`
replicates the fitness unconditionally, so both sides pay the same gather
(docs/03, "no longer measurable by difference"). The two honest fixes are
making the replicate conditional on the shaping, which is a core.py protocol
change `tell`'s docstring argues against, or measuring the barrier on its own.
This is the second one.

The barrier in `tell` is exactly two steps and this times them separately:

  gather  `with_sharding_constraint(fitness, replicated)`   -> shaping "none"
  sort    `centered_ranks` on the replicated array          -> the increment

Per (N, D, shaping) cell the timed function is that barrier alone, on a
fitness pinned to the same `P("pop")` sharding `apply` leaves it in. Chained
inside `lax.scan` (each iteration's fitness is perturbed by the previous
weights) because a single ~10 us op timed from Python measures dispatch, not
the op; the scan is timed whole and divided by its length. `D=1` rows are the
no-communication floor, so t(D) - t(1) is the communication share and
t(centered_ranks) - t(none) at the same (N, D) is the sort.

What this does not measure: overlap. Inside a real generation the compiler can
hide some of this latency behind the contraction; a standalone number is the
barrier's worst case, and the honest comparison in the paper is this ceiling
next to M5's in-context generation times, not instead of them.
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

from shardes import shaping as shapings  # noqa: E402
from shardes import sharding  # noqa: E402

OUTPUTS = ("results-barrier",)
CHAIN = 100  # scan length; long enough that per-iteration dispatch is amortized away


@dataclass(frozen=True)
class Config:
    population: int
    devices: int
    shaping: str  # "none" | "centered_ranks"

    def slug(self) -> str:
        return f"N={self.population}__D={self.devices}__s={self.shaping}"


def expand(cfg: dict) -> list[Config]:
    return [
        Config(n, d, s)
        for n in cfg["population"]
        for d in cfg["devices"]
        for s in cfg["shaping"]
        if n % d == 0
    ]


def measure(config: Config, cfg: dict) -> dict:
    warmup = int(cfg.get("warmup", 3))
    repeats = int(cfg.get("repeats", 5))

    mesh = sharding.make_mesh(config.devices)
    shape_fn = shapings.BY_NAME[config.shaping]
    replicated = sharding.replicated(mesh)
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("pop"))

    # The fitness `apply` hands to `tell`: f32 scalars, sharded over the member
    # axis. Values are arbitrary but distinct, so the sort does real work.
    fitness = jax.device_put(
        jax.random.normal(jax.random.key(0), (config.population,), jnp.float32),
        sharded,
    )

    @jax.jit
    def chain(f):
        def step(f, _):
            weights = shape_fn(jax.lax.with_sharding_constraint(f, replicated))
            # Feed the output back in, re-pinned to the input sharding, so the
            # scan cannot hoist the barrier out of the loop. The perturbation
            # keeps every iteration's values distinct.
            f = jax.lax.with_sharding_constraint(f + 1e-6 * weights, sharded)
            return f, None
        f, _ = jax.lax.scan(step, f, None, length=CHAIN)
        return f

    for _ in range(warmup):
        jax.block_until_ready(chain(fitness))
    seconds = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(chain(fitness))
        seconds.append((time.perf_counter() - t0) / CHAIN)

    return {
        "config": config.__dict__,
        "chain_length": CHAIN,
        "seconds_all": seconds,
        "seconds_median": statistics.median(seconds),
        "seconds_iqr": [sorted(seconds)[len(seconds) // 4],
                        sorted(seconds)[-1 - len(seconds) // 4]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=float, default=None,
                    help="stop scheduling once the session has used this many seconds")
    args = ap.parse_args(argv)
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(args.config.read_text())
    results = HERE / cfg.get("results_dir", "results-barrier")
    configs = expand(cfg)
    available = jax.device_count()
    runnable = [c for c in configs if c.devices <= available]
    pending = [c for c in runnable if not (results / f"{c.slug()}.json").exists()]
    print(f"{len(configs)} configs, {len(runnable)} runnable on {available} "
          f"device(s), {len(pending)} pending")
    if args.dry_run:
        for c in pending:
            print(" ", c.slug())
        return 0

    results.mkdir(exist_ok=True)
    env = harness.capture_env(HERE, (*OUTPUTS, cfg.get("results_dir", "results-barrier")))
    done = 0
    started = time.perf_counter()
    stopped_early = None
    for i, c in enumerate(pending):
        elapsed = time.perf_counter() - started
        if args.budget is not None and elapsed > args.budget:
            stopped_early = (i, elapsed)
            break
        rec = measure(c, cfg)
        rec["env"] = env
        (results / f"{c.slug()}.json").write_text(json.dumps(rec, indent=1))
        done += 1
        print(f"[{i + 1}/{len(pending)}] {c.slug()}  "
              f"{rec['seconds_median'] * 1e6:.1f} us/iter", flush=True)

    print(f"done: {done} measured")
    if stopped_early:
        i, elapsed = stopped_early
        print(f"STOPPED at {i}/{len(pending)} after {elapsed / 3600:.2f} h "
              f"(budget {args.budget / 3600:.2f} h). Re-run to continue; results resume.")
        return 1
    if len(runnable) < len(configs):
        print("EXITING NON-ZERO: configurations skipped for needing more devices.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
