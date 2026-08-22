#!/usr/bin/env python
"""Isolated collective timings on the full local mesh (review 7, hole 2).

    python allreduce_ladder.py                 # all local devices
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      JAX_PLATFORMS=cpu python allreduce_ladder.py --smoke

Times, in isolation and with the sweeps' discipline (warm-up, fenced
repeats, median and IQR), the two collectives the contraction placements
issue: strategy B's model-sized all-reduce, at 8 B, 1 KB, 1 MB, the float32
parameter payload of the d=512 and d=2048 blocks (6d^2 floats) and 100 MiB;
and the 4N-byte fitness all-gather both placements share, at N in {256,
1024, 2^18}. Fits alpha (latency) and beta (bandwidth) from the 8 B and
100 MiB all-reduce points. One JSON per platform under results-ladder/,
stamped with the environment, so the crossover's end-to-end ratios can be
read beside the isolated cost of the transfer they contain.
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
from jax.sharding import NamedSharding, PartitionSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import harness  # noqa: E402
from shardes import sharding  # noqa: E402

ALLREDUCE_BYTES = [8, 1024, 2**20, 6 * 512**2 * 4, 6 * 2048**2 * 4, 100 * 2**20]
ALLGATHER_N = [256, 1024, 2**18]


def timed(fn, x, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn(x).block_until_ready()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        ts.append(time.perf_counter() - t0)
    s = sorted(ts)
    return {"median": statistics.median(ts), "all": ts,
            "iqr": [s[len(s) // 4], s[-1 - len(s) // 4]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny payloads only")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args(argv)

    devs = jax.devices()
    d_count = len(devs)
    mesh = sharding.make_mesh(d_count)
    rep = NamedSharding(mesh, PartitionSpec())
    member = NamedSharding(mesh, PartitionSpec(sharding.POP))
    out_dir = HERE / "results-ladder"
    out_dir.mkdir(exist_ok=True)
    kind = devs[0].device_kind.replace(" ", "-").lower()
    out = out_dir / f"ladder-{kind}-D{d_count}{'-smoke' if args.smoke else ''}.json"
    env = harness.capture_env(HERE, ("results-ladder",))

    rec = {"devices": d_count, "device_kind": devs[0].device_kind,
           "platform": devs[0].platform, "env": env,
           "allreduce": {}, "allgather": {}}

    sizes = ALLREDUCE_BYTES[:3] if args.smoke else ALLREDUCE_BYTES
    for size_bytes in sizes:
        n = max(size_bytes // 4, 2)
        # B's all-reduce as the library issues it: a member-sharded partial
        # summed over the population axis into a replicated result.
        y = jax.device_put(jnp.ones((d_count, n // d_count + 1), jnp.float32), member)

        @jax.jit
        def psum_like(v):
            return jax.lax.with_sharding_constraint(v.sum(axis=0), rep)

        r = timed(psum_like, y, args.warmup, args.repeats)
        rec["allreduce"][str(size_bytes)] = r
        print(f"all-reduce {size_bytes:>11d} B: {r['median'] * 1e6:10.1f} us", flush=True)

    for n in (ALLGATHER_N[:2] if args.smoke else ALLGATHER_N):
        # The fitness gather: N scalars sharded over the population, replicated.
        x = jax.device_put(jnp.arange(n, dtype=jnp.float32), member)

        @jax.jit
        def gather(v):
            return jax.lax.with_sharding_constraint(v, rep)

        r = timed(gather, x, args.warmup, args.repeats)
        rec["allgather"][str(n)] = r
        print(f"all-gather N={n:>7d} ({4 * n} B): {r['median'] * 1e6:10.1f} us", flush=True)

    if not args.smoke:
        small = rec["allreduce"]["8"]["median"]
        big = rec["allreduce"][str(100 * 2**20)]["median"]
        rec["alpha_seconds"] = small
        rec["beta_bytes_per_second"] = (100 * 2**20) / max(big - small, 1e-9)
        print(f"alpha ~{small * 1e6:.0f} us, beta ~{rec['beta_bytes_per_second'] / 2**30:.1f} GiB/s")
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
