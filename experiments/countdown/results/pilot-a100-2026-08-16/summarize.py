#!/usr/bin/env python
"""Reproduce the README table from the arm logs in this directory."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

print(f"{'arm':16s} {'gens':>4s} {'med s/gen':>9s} {'total min':>9s} "
      f"{'reward':>7s} {'solved':>7s} {'format':>7s}   (last-50-gen means)")
for name in ["mirrored-seed", "lr1", "lr4", "lr16"]:
    xs = [json.loads(l) for l in (HERE / f"arm-{name}.jsonl").open()]
    steady = [x["seconds"] for x in xs if x["generation"] >= 2]
    last50 = [x for x in xs if x["generation"] >= len(xs) - 50]
    print(f"{name:16s} {len(xs):4d} {statistics.median(steady):9.1f} "
          f"{sum(x['seconds'] for x in xs) / 60:9.1f} "
          f"{statistics.mean(x['reward_mean'] for x in last50):7.3f} "
          f"{statistics.mean(x['solved_frac'] for x in last50):7.3f} "
          f"{statistics.mean(x['format_frac'] for x in last50):7.2f}")
