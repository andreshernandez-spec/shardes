#!/usr/bin/env python
"""Where seed regeneration's 5-6x on the v5e goes (review 7, hole 2).

    python regen_decompose.py                  # one device, cost-surface cells
    JAX_PLATFORMS=cpu python regen_decompose.py --smoke

The cost surface (results-cost*, cost.py) measured seed_regenerated at
5-6x the materializing dense baseline per device on the TPU against
1.3-3x on the A100, end to end. Both arms apply a full-rank perturbation
per member, so neither keeps a shared base matmul; what differs is that
the regenerating arm draws its noise twice per generation (evaluate and
contract) instead of reading it from memory once. This script times, at
the cost surface's own cells and discipline (cost.measure, bfloat16
compute, default matmul precision), three things per cell:

  t_iid   one generation, materialized i.i.d. noise
  t_seed  one generation, seed-regenerated noise
  t_rng   drawing N x P float32 normals with the library's counter RNG,
          member by member under a scan (no N x P array ever exists)

and reports (t_seed - t_iid) / (2 t_rng). A ratio near one says the gap is
random-number throughput; well above one says the regenerating program
costs more than its extra draws (memory traffic, fusion, the two passes);
well below one says regeneration overlaps with something. One JSON per
cell under results-regen/, stamped with the environment.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import harness  # noqa: E402
from cost import Config, measure  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402

CELLS = [(512, 256), (512, 1024), (2048, 256), (2048, 1024)]


def rng_seconds(n: int, p: int, warmup: int, repeats: int) -> float:
    key = jax.random.key(0)

    @jax.jit
    def draw_all(k):
        def body(acc, i):
            z = jax.random.normal(jax.random.fold_in(k, i), (p,), jnp.float32)
            return acc + z[0] + z[-1], None  # touch the draw; never keep it
        acc, _ = jax.lax.scan(body, jnp.float32(0), jnp.arange(n))
        return acc

    for _ in range(warmup):
        draw_all(key).block_until_ready()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        draw_all(key).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = {"warmup": 3, "repeats": 5, "matmul_precision": "default",
           "batch": 8, "seq": 32}
    cells = [(64, 8), (64, 16)] if args.smoke else CELLS
    out_dir = HERE / ("results-regen-smoke" if args.smoke else "results-regen")
    out_dir.mkdir(exist_ok=True)
    env = harness.capture_env(HERE, (out_dir.name,))
    kind = jax.devices()[0].device_kind.replace(" ", "-").lower()

    for d, n in cells:
        out = out_dir / f"d={d}__N={n}__{kind}.json"
        if out.exists():
            print(f"skip {out.name}")
            continue
        p = sum(x.size for x in jax.tree.leaves(
            transformer_block.init(jax.random.key(0), d_model=d)))
        rec = {"d_model": d, "population": n, "params": int(p), "env": env,
               "device_kind": jax.devices()[0].device_kind}
        try:
            for arm in ("iid_gaussian", "seed_regenerated"):
                r = measure(Config(d, n, arm, "bfloat16"), cfg, out_dir)
                rec[f"t_{'iid' if arm == 'iid_gaussian' else 'seed'}"] = r["seconds_median"]
            rec["t_rng"] = rng_seconds(n, int(p), cfg["warmup"], cfg["repeats"])
            gap = rec["t_seed"] - rec["t_iid"]
            rec["gap_over_two_rng"] = gap / (2 * rec["t_rng"])
            print(f"d={d} N={n}: iid {rec['t_iid'] * 1e3:.1f} ms, seed {rec['t_seed'] * 1e3:.1f} ms, "
                  f"rng {rec['t_rng'] * 1e3:.1f} ms, (seed-iid)/(2 rng) = {rec['gap_over_two_rng']:.2f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            if "RESOURCE_EXHAUSTED" not in msg:
                raise
            rec["status"] = "oom"
            rec["error"] = msg[:300]
            print(f"d={d} N={n}: OOM (recorded)", flush=True)
        out.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
