#!/usr/bin/env python
"""Reproduce the README tables from the eval logs in this directory."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def final(path):
    return [json.loads(l) for l in path.open()][-1]


print("ES arms, final held-out eval over seeds 0-2 (mean [min-max]):")
for arm in ["mirrored-seed", "mirrored-lr1", "mirrored-lr4", "mirrored-lr16"]:
    rew, sol = [], []
    for s in (0, 1, 2):
        x = final(HERE / f"es-{arm}-s{s}-eval.jsonl")
        rew.append(x["eval_reward"])
        sol.append(x["eval_solved"])
    print(f"  {arm:14s} reward {statistics.mean(rew):.3f} [{min(rew):.3f}-{max(rew):.3f}]"
          f"   solved {statistics.mean(sol):.3f} [{min(sol):.3f}-{max(sol):.3f}]")

print("GRPO, final held-out eval per seed:")
for s in (0, 1, 2):
    x = final(HERE / f"grpo-s{s}-eval.jsonl")
    print(f"  seed {s}: reward {x['eval_reward']:.3f}   solved {x['eval_solved']:.3f}"
          f"   format {x['eval_format']:.2f}")

base = [json.loads(l) for l in (HERE / "es-mirrored-seed-s0-eval.jsonl").open()][0]
print(f"base model (ES decoder): reward {base['eval_reward']:.3f} "
      f"solved {base['eval_solved']:.3f} format {base['eval_format']:.2f}")
