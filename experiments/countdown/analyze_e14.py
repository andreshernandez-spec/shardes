#!/usr/bin/env python
"""E14 verdicts from the committed runs. Markdown table to stdout, no hand-edits.

    python analyze_e14.py

Metrics exactly as docs/08 defined them before the data: final held-out reward at
the 150-step horizon; collapsed = final below the arm's own decoder floor;
frozen = less than a quarter of the multiplier-1 reference progress; drawdown =
max peak-to-later-trough of the eval curve; drift = parameter L2 from the
pretrained weights where the run recorded it (E13's x1 runs predate the metric).

The x1 columns are E13's committed curves read at the same horizon
(results/e13-a100-2026-08-17, generation/step <= 150), which is what anchors
E14 to the published-settings baseline.
"""

from __future__ import annotations

import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
E14 = HERE / "results" / "e14-a100-2026-08-20"
E13 = HERE / "results" / "e13-a100-2026-08-17"
HORIZON = 150

#: Decoder-family base floors, E13's cross-decoder rule: each family against its
#: own generation-0 floor (JAX greedy vs HF greedy differ by bf16 padding).
FLOOR = {"es": 0.054, "grpo": 0.037}


def curve(path: pathlib.Path):
    rows = [json.loads(l) for l in path.open()]
    key = "generation" if "generation" in rows[0] else "step"
    return [(r[key], r["eval_reward"], r.get("param_l2_from_init")) for r in rows
            if r[key] <= HORIZON]


def verdict(family: str, pts, ref_progress: float):
    floor = FLOOR[family]
    final = pts[-1][1]
    peak = 0.0
    drawdown = 0.0
    for _, r, _ in pts:
        peak = max(peak, r)
        drawdown = max(drawdown, peak - r)
    collapsed = final < floor
    frozen = (not collapsed) and (final - floor) < 0.25 * ref_progress
    return final, collapsed, frozen, drawdown, pts[-1][2]


def main() -> int:
    # multiplier-1 reference progress per family, from E13 at the same horizon
    ref = {}
    for family, stem in (("es", "es-mirrored-lr1"), ("grpo", "grpo")):
        finals = [curve(E13 / f"{stem}-s{s}-eval.jsonl")[-1][1] for s in (0, 1, 2)]
        ref[family] = statistics.median(finals) - FLOOR[family]

    cells = [
        ("es", "es-lr-eighth"), ("es", "es-lr-8x"),
        ("es", "es-sigma-quarter"), ("es", "es-sigma-4x"),
        ("grpo", "grpo-lr-eighth"), ("grpo", "grpo-lr-8x"),
        ("grpo", "grpo-beta-0"), ("grpo", "grpo-clip-off"),
    ]
    print("| cell | seed | final | collapsed | frozen | drawdown | drift |")
    print("|---|---|---|---|---|---|---|")
    counts = {"es": [0, 0], "grpo": [0, 0]}  # [collapsed, frozen]
    for family, cell in cells:
        for s in (0, 1, 2):
            pts = curve(E14 / f"{cell}-s{s}" / "eval.jsonl")
            if family == "grpo":  # drift lives in summary.json for GRPO
                summ = json.loads((E14 / f"{cell}-s{s}" / "summary.json").read_text())
                pts[-1] = (*pts[-1][:2], summ["param_l2_from_init"])
            final, coll, froz, dd, drift = verdict(family, pts, ref[family])
            counts[family][0] += coll
            counts[family][1] += froz
            print(f"| {cell} | {s} | {final:.3f} | {'YES' if coll else ''} | "
                  f"{'YES' if froz else ''} | {dd:.3f} | "
                  f"{'-' if drift is None else f'{drift:.1f}'} |")
    print()
    for family in ("es", "grpo"):
        c, f = counts[family]
        print(f"{family}: {c} collapsed, {f} frozen of 12 perturbed runs "
              f"(x1 reference progress {ref[family]:.3f})")
    # x1 drift baseline (es-x1, run 2026-08-21 at the E14 horizon with the
    # metric): E13's x1 runs predate param_l2_from_init, so the drift column's
    # baseline comes from these three replication runs. Their finals double as
    # an independent reproduction of E13's x1 arm.
    print("\nx1 drift baseline (es-x1):")
    for s in (0, 1, 2):
        pts = curve(E14 / f"es-x1-s{s}" / "eval.jsonl")
        print(f"  s{s}: final {pts[-1][1]:.3f}, drift {pts[-1][2]:.2f}")

    # E13 published-settings runs at the same horizon, for the combined count
    print("\nE13 x1 columns at the same horizon:")
    for family, stem in (("es", "es-mirrored-lr1"), ("grpo", "grpo")):
        for s in (0, 1, 2):
            pts = curve(E13 / f"{stem}-s{s}-eval.jsonl")
            final, coll, froz, dd, _ = verdict(family, pts, ref[family])
            marks = ("COLLAPSED" if coll else "frozen" if froz else "ok")
            print(f"  {stem}-s{s}: final {final:.3f} {marks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
