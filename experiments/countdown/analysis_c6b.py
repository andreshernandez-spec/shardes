#!/usr/bin/env python
"""C6b: does F5's estimator-quality ordering predict the E13 task outcome?

Reads only committed artifacts: E1's estimator-quality results
(experiments/phase0/results) and E13's held-out eval logs
(results/e13-a100-2026-08-17). No GPU, no network.

Method. For each E13 arm with an E1 curve (full, r=1, r=4; E1 never measured
r=16), take the E1 points at sigma matching the E13 config (1e-3), fit
log cos = a + b log(N/d_eff) over the arm's own curve, and evaluate the fit at
the arm's E13 operating point: N = 30 members on the Qwen2.5-0.5B tree, with
d_eff from shardes.dimensions.sampling_dimension over the real parameter shapes
(jax.eval_shape, nothing materialized). That gives F5's predicted estimator
quality, and with it a predicted ordering. The observed ordering is the final
held-out eval over seeds 0-2. Both shaping slices are reported: `none` is F5's
baseline (the slice where cos measures gradient estimation), `centered_ranks`
is what E13's tell actually applies.
"""
import json
import math
import statistics
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "phase0" / "results"
E13 = HERE / "results" / "e13-a100-2026-08-17"

import sys

sys.path.insert(0, str(HERE.parent))
from shardes.dimensions import FULL, sampling_dimension  # noqa: E402
from shardes.problems import qwen2  # noqa: E402

SIGMA = "1e-03"          # matches E13's sigma 0.001
N_E13 = 30

cfg = qwen2.Config.qwen25_05b()
tree = jax.eval_shape(lambda: qwen2.init(jax.random.key(0), cfg, dtype=jnp.bfloat16))

ARMS = {  # E13 arm -> (E1 strategy, rank for d_eff, E13 eval-log stem)
    "full": ("mirrored_full", FULL, "es-mirrored-seed"),
    "r1": ("mirrored_lr1", 1, "es-mirrored-lr1"),
    "r4": ("mirrored_lr4", 4, "es-mirrored-lr4"),
}


def e1_curve(strategy, shaping):
    pts = []
    for f in E1.glob(f"strategy={strategy}__N=*__sigma={SIGMA}__shaping={shaping}.json"):
        d = json.loads(f.read_text())
        pts.append((d["config"]["population"] / d["d_eff"], d["cosine_median"]))
    return sorted(pts)


def fit_predict(pts, x):
    lx = [math.log(p[0]) for p in pts]
    ly = [math.log(p[1]) for p in pts]
    n = len(pts)
    mx, my = sum(lx) / n, sum(ly) / n
    b = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    a = my - b * mx
    return math.exp(a + b * math.log(x)), b


def observed(stem):
    finals = []
    for s in (0, 1, 2):
        rows = [json.loads(l) for l in (E13 / f"{stem}-s{s}-eval.jsonl").open()]
        finals.append(rows[-1]["eval_reward"])
    return statistics.mean(finals), min(finals), max(finals)


print(f"E13 operating point: N = {N_E13} on Qwen2.5-0.5B "
      f"({sampling_dimension(tree, FULL):,} params)\n")
for shaping in ("none", "centered_ranks"):
    print(f"shaping = {shaping}" + ("   (F5's baseline slice)" if shaping == "none"
                                    else "   (what E13's tell applies)"))
    rows = []
    for arm, (strategy, rank, stem) in ARMS.items():
        d_eff = sampling_dimension(tree, rank)
        x = N_E13 / d_eff
        pts = e1_curve(strategy, shaping)
        pred, slope = fit_predict(pts, x)
        mean, lo, hi = observed(stem)
        rows.append((arm, d_eff, x, pred, slope, mean, lo, hi))
    for arm, d_eff, x, pred, slope, mean, lo, hi in rows:
        print(f"  {arm:5s} d_eff {d_eff:>11,}  N/d_eff {x:8.1e}  "
              f"predicted cos {pred:8.1e} (slope {slope:+.2f})  "
              f"observed eval {mean:.3f} [{lo:.3f}-{hi:.3f}]")
    best, worst = max(rows, key=lambda r: r[3]), min(rows, key=lambda r: r[3])
    print(f"  F5 predicts {best[0]} over {worst[0]} by "
          f"{best[3] / worst[3]:.0f}x in estimator quality; observed final "
          f"evals are a statistical tie.\n")
