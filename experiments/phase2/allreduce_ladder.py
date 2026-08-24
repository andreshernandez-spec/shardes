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
1024, 2^18}. The payload is what each device holds: for the all-reduce
every device contributes one partial of the full size, as strategy B's
psum does (contraction.contract_sharded).

Two numbers per size. `call` is one jitted program holding the one
collective, timed from Python: dispatch, the transfer, and the sync, which
is the floor a collective pays when it is the whole program. `step` is the
cost of one more collective inside a program that already runs, from the
slope of a dependent chain of K collectives (K in {1, 9}) minus the slope
of the same chain without the collective. That is the cost the ES loop
pays, where the psum sits inside the generation program. Each chain step
adds a per-device term, so XLA cannot prove the operand replicated and
drop the all-reduce. Alpha (latency) and beta (bandwidth) are fitted from
the 8 B and 100 MiB step costs. One JSON per platform under
results-ladder/, stamped with the environment.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import harness  # noqa: E402
from shardes import sharding  # noqa: E402

POP = sharding.POP
ALLREDUCE_BYTES = [8, 1024, 2**20, 6 * 512**2 * 4, 6 * 2048**2 * 4, 100 * 2**20]
ALLGATHER_N = [256, 1024, 2**18]
CHAIN = (1, 9)


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


def allreduce_programs(mesh, d_count: int, n: int) -> dict:
    """`call`: one psum. `chain[k]`: k dependent psums. `nocomm[k]`: the same loop without."""
    inv = jnp.float32(1.0 / d_count)

    def body(comm: bool):
        def step(c, _):
            s = jax.lax.psum(c, POP) if comm else c
            # The device index keeps the carry unreplicated, so the next psum is real work.
            idx = jax.lax.axis_index(POP).astype(jnp.float32)
            return s * inv + idx * jnp.float32(1e-9), None
        return step

    def make(comm: bool, k: int):
        def local(v):
            c, _ = jax.lax.scan(body(comm), v, None, length=k)
            return c
        return jax.jit(shard_map(local, mesh=mesh, in_specs=P(POP), out_specs=P(POP),
                                 check_vma=False))

    def call_local(v):
        return jax.lax.psum(v, POP)

    call = jax.jit(shard_map(call_local, mesh=mesh, in_specs=P(POP), out_specs=P(),
                             check_vma=False))
    progs = {"call": call}
    for k in CHAIN:
        progs[f"chain{k}"] = make(True, k)
        progs[f"nocomm{k}"] = make(False, k)
    return progs


def allgather_programs(mesh, d_count: int, n: int) -> dict:
    local_n = n // d_count

    def body(comm: bool):
        def step(c, _):
            g = jax.lax.all_gather(c, POP, tiled=True) if comm else jnp.tile(c, d_count)
            # Each device takes its own slice back, so the carry stays unreplicated.
            start = jax.lax.axis_index(POP) * local_n
            return jax.lax.dynamic_slice_in_dim(g, start, local_n) + jnp.float32(1e-9), None
        return step

    def make(comm: bool, k: int):
        def local(v):
            c, _ = jax.lax.scan(body(comm), v, None, length=k)
            return c
        return jax.jit(shard_map(local, mesh=mesh, in_specs=P(POP), out_specs=P(POP),
                                 check_vma=False))

    def call_local(v):
        return jax.lax.all_gather(v, POP, tiled=True)

    call = jax.jit(shard_map(call_local, mesh=mesh, in_specs=P(POP), out_specs=P(),
                             check_vma=False))
    progs = {"call": call}
    for k in CHAIN:
        progs[f"chain{k}"] = make(True, k)
        progs[f"nocomm{k}"] = make(False, k)
    return progs


def measure(progs: dict, x, warmup: int, repeats: int) -> dict:
    r = {name: timed(fn, x, warmup, repeats) for name, fn in progs.items()}
    k0, k1 = CHAIN
    slope = (r[f"chain{k1}"]["median"] - r[f"chain{k0}"]["median"]) / (k1 - k0)
    slope0 = (r[f"nocomm{k1}"]["median"] - r[f"nocomm{k0}"]["median"]) / (k1 - k0)
    return {"call": r["call"], "chain": {str(k): r[f"chain{k}"] for k in CHAIN},
            "nocomm": {str(k): r[f"nocomm{k}"] for k in CHAIN},
            "step_seconds": slope - slope0, "step_raw_seconds": slope,
            "step_nocomm_seconds": slope0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny payloads only")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args(argv)

    devs = jax.devices()
    d_count = len(devs)
    mesh = sharding.make_mesh(d_count)
    member = NamedSharding(mesh, P(POP))
    out_dir = HERE / "results-ladder"
    out_dir.mkdir(exist_ok=True)
    kind = devs[0].device_kind.replace(" ", "-").lower()
    out = out_dir / f"ladder-{kind}-D{d_count}{'-smoke' if args.smoke else ''}.json"
    env = harness.capture_env(HERE, ("results-ladder", "results-regen"))

    rec = {"devices": d_count, "device_kind": devs[0].device_kind,
           "platform": devs[0].platform, "env": env, "chain": list(CHAIN),
           "allreduce": {}, "allgather": {}}

    sizes = ALLREDUCE_BYTES[:3] if args.smoke else ALLREDUCE_BYTES
    for size_bytes in sizes:
        n = max(size_bytes // 4, 2)
        # Every device holds a full-size partial, as B's contraction leaves it.
        y = jax.device_put(jnp.ones((d_count, n), jnp.float32), member)
        r = measure(allreduce_programs(mesh, d_count, n), y, args.warmup, args.repeats)
        rec["allreduce"][str(size_bytes)] = r
        print(f"all-reduce {size_bytes:>11d} B: call {r['call']['median'] * 1e6:9.1f} us, "
              f"step {r['step_seconds'] * 1e6:9.1f} us", flush=True)

    for n in (ALLGATHER_N[:2] if args.smoke else ALLGATHER_N):
        # The fitness gather: N scalars sharded over the population, then replicated.
        x = jax.device_put(jnp.arange(n, dtype=jnp.float32), member)
        r = measure(allgather_programs(mesh, d_count, n), x, args.warmup, args.repeats)
        rec["allgather"][str(n)] = r
        print(f"all-gather N={n:>7d} ({4 * n} B): call {r['call']['median'] * 1e6:9.1f} us, "
              f"step {r['step_seconds'] * 1e6:9.1f} us", flush=True)

    if not args.smoke:
        small = rec["allreduce"]["8"]["step_seconds"]
        big = rec["allreduce"][str(100 * 2**20)]["step_seconds"]
        rec["alpha_seconds"] = small
        rec["beta_bytes_per_second"] = (100 * 2**20) / max(big - small, 1e-9)
        rec["call_floor_seconds"] = rec["allreduce"]["8"]["call"]["median"]
        print(f"alpha ~{small * 1e6:.0f} us, beta ~{rec['beta_bytes_per_second'] / 2**30:.1f} GiB/s, "
              f"call floor ~{rec['call_floor_seconds'] * 1e6:.0f} us")
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
