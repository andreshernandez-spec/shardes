#!/usr/bin/env python
"""E15 analysis: measured real-model cosines against F5's fitted predictions.

    python analysis_e15.py                     # reads results-e15/
    python analysis_e15.py --results DIR       # e.g. a partial fetch

Reads only committed artifacts: E1's estimator-quality results
(experiments/phase0/results) and E15's per-cell JSONs. No GPU, no network.

The bridge being tested: F5 fitted log cos = a + b log(N/d_eff) on synthetic
objectives; E15 measured the same cosine on Qwen2.5-0.5B's own NLL surface.
If the fit evaluated at E15's operating points lands near the measured
medians, the synthetic curves transfer to the real model and C6b's use of
them is grounded. The fit machinery is analysis_c6b's, unchanged.

E15's update went through ShardedES's default shaping (centered_ranks), so
that slice is the like-for-like prediction; the `none` slice is printed too
because it is F5's baseline. E1 never measured r=16, so mirrored_lr16 gets
a measured row and no prediction. The unpaired lowrank_r1 has its own E1
curve, so the arm EGGROLL's sampler cannot express is predicted too.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "phase0" / "results"

sys.path.insert(0, str(HERE.parent))
from shardes.dimensions import FULL, sampling_dimension  # noqa: E402
from shardes.problems import qwen2  # noqa: E402

SIGMA = "1e-03"  # matches e15.yaml's sigma 0.001

ARMS = {  # E15 strategy -> (E1 strategy or None, rank for d_eff)
    "mirrored_seed": ("mirrored_full", FULL),
    "mirrored_lr1": ("mirrored_lr1", 1),
    "mirrored_lr4": ("mirrored_lr4", 4),
    "mirrored_lr16": (None, 16),  # E1 never measured r=16
    "lowrank_r1": ("lowrank_r1", 1),
}


def e1_curve(strategy: str, shaping: str):
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
    return math.exp(a + b * math.log(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results-e15")
    ap.add_argument("--latex", action="store_true",
                    help="also write paper/generated/tb4.tex (like-for-like slice)")
    args = ap.parse_args(argv)

    cfg = qwen2.Config.qwen25_05b()
    tree = jax.eval_shape(lambda: qwen2.init(jax.random.key(0), cfg,
                                             dtype=jnp.bfloat16))
    cells = {}
    for f in sorted(args.results.glob("s=*__N=*.json")):
        d = json.loads(f.read_text())
        cells[(d["config"]["strategy"], d["config"]["population"])] = d

    if not cells:
        print(f"no cells in {args.results}")
        return 1

    print(f"E15 operating points on Qwen2.5-0.5B "
          f"({sampling_dimension(tree, FULL):,} params); "
          f"{len(cells)} measured cells\n")
    for shaping in ("centered_ranks", "none"):
        print(f"shaping = {shaping}"
              + ("   (what E15's tell applied; the like-for-like slice)"
                 if shaping == "centered_ranks" else "   (F5's baseline slice)"))
        for (strategy, n), d in sorted(cells.items()):
            e1_name, rank = ARMS[strategy]
            d_eff = sampling_dimension(tree, rank)
            cos = sorted(d["cosines"])
            med = d["cosine_median"]
            line = (f"  {strategy:13s} N={n:<4d} d_eff {d_eff:>11,}  "
                    f"measured {med:8.1e} [{cos[0]:.1e}-{cos[-1]:.1e}]")
            if e1_name is None:
                print(line + "  (no E1 curve for r=16; measured only)")
                continue
            pts = e1_curve(e1_name, shaping)
            if not pts:
                print(line + f"  (no E1 points for {e1_name} at this shaping)")
                continue
            pred = fit_predict(pts, n / d_eff)
            print(line + f"  predicted {pred:8.1e}  ratio {med / pred:4.2f}x")
        print()

    if args.latex:
        names = {"mirrored_seed": "full rank", "mirrored_lr1": "rank 1",
                 "mirrored_lr4": "rank 4", "mirrored_lr16": "rank 16",
                 "lowrank_r1": "rank 1, unpaired"}
        order = ["mirrored_seed", "mirrored_lr1", "mirrored_lr4",
                 "mirrored_lr16", "lowrank_r1"]
        rows = []
        for strategy in order:
            for n in (30, 240):
                d = cells[(strategy, n)]
                e1_name, rank = ARMS[strategy]
                d_eff = sampling_dimension(tree, rank)
                cos = sorted(d["cosines"])
                meas = (f"${d['cosine_median'] * 1e4:.1f}$ "
                        f"[{cos[0] * 1e4:.1f}, {cos[-1] * 1e4:.1f}]")
                if e1_name is None:
                    pred, ratio = "--", "--"
                else:
                    pv = fit_predict(e1_curve(e1_name, "centered_ranks"), n / d_eff)
                    pred = f"${pv * 1e4:.1f}$"
                    ratio = f"${d['cosine_median'] / pv:.2f}\\times$"
                rows.append(f"{names[strategy]} & {n} & {meas} & {pred} & {ratio} \\\\")
        out = Path(__file__).resolve().parent.parent.parent / "paper" / "generated" / "tb4.tex"
        out.write_text("\n".join([
            "% generated by experiments/countdown/analysis_e15.py --latex; do not edit",
            "\\begin{table*}[t]", "\\centering", "\\small",
            "\\caption{The synthetic-to-real bridge, measured. Cosine between one",
            "  production update and the true NLL gradient on Qwen2.5-0.5B",
            "  ($\\times 10^{-4}$; median [min, max] over 5 seeds), against the",
            "  estimator study's fitted prediction at the same $N/d_{\\mathrm{eff}}$",
            "  (centered-rank slice, the shaping the update applies). The estimator",
            "  study has no rank-16 curve, so that arm is measured only.}",
            "\\label{tab:tb4}",
            "\\begin{tabular}{llllr}", "\\toprule",
            "arm & $N$ & measured & predicted & ratio \\\\", "\\midrule",
            *rows,
            "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
