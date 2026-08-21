#!/usr/bin/env python
"""Regenerate figure F7 from results/e13-a100-2026-08-17. No hand-edited numbers.

    python plot_e13.py        # figures/f7-e13-heldout.png

F7: held-out eval reward vs sample evaluations, one curve per arm, min-max band
over seeds 0-2. Four ES arms, the GRPO reference, the frozen-embedding ablation,
and the base model as a dashed floor. The x-axis is sample evaluations, not
generations or steps, because that is the budget Qiu's protocol matches: one ES
generation is N * puzzles_per_gen = 240, one GRPO step is prompts * group = 240,
so the two proceed at the same rate and the axis is honest for both.

The claims the figure carries: the four ES curves sit on top of each other
(C6a: the rank axis is flat on held-out quality), the frozen-embedding curve
sits with them (C6c), and the GRPO band is wide where the ES bands are tight
(the seed-2 collapse to format-only is inside it).

The two zero points differ (ES 0.054, GRPO 0.037 on the same base weights):
that is the residual cross-decoder delta the campaign README quantifies, bf16
right-vs-left padding numerics, and it bounds how small a cross-arm gap can be
read off this figure. Within-family comparisons share one decoder.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this has to work over ssh and in a notebook driver
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "e13-a100-2026-08-17"
FIGURES = HERE / "figures"

EVALS_PER_UNIT = 240  # per ES generation and per GRPO step alike; see module docstring

# Colour per arm, fixed so the same arm is the same colour in every figure and in any
# talk that reuses them. The ES family shades one hue by rank; GRPO is the one red.
ARM_STYLE = {
    "es-mirrored-seed": ("#08306b", "full rank (Qiu)"),
    "es-mirrored-lr1": ("#2171b5", "rank 1"),
    "es-mirrored-lr4": ("#6baed6", "rank 4"),
    "es-mirrored-lr16": ("#a6bddb", "rank 16"),
    "es-lr1-frozen-embed": ("#2ca02c", "rank 1, embedding frozen"),
    "grpo": ("#d62728", "GRPO (Qiu's settings)"),
}


def curves(stem: str, xkey: str):
    """[(x_evals, [reward per seed]) ...] over the arm's three seed files."""
    by_x = {}
    for s in (0, 1, 2):
        for row in map(json.loads, (RESULTS / f"{stem}-s{s}-eval.jsonl").open()):
            by_x.setdefault(row[xkey] * EVALS_PER_UNIT, []).append(row["eval_reward"])
    return sorted(by_x.items())


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # Both decoders' base-model floors, labeled: the JAX evaluator (ES arms)
    # and the HF evaluator (GRPO) score the same weights differently by a
    # bf16 padding artifact, and an unlabeled single floor invites misreading
    # the gap between them as a training effect.
    for stem, label in (("es-mirrored-seed", "base model (JAX decoder)"),
                        ("grpo", "base model (HF decoder)")):
        floor = json.loads((RESULTS / f"{stem}-s0-eval.jsonl")
                           .open().readline())["eval_reward"]
        ax.axhline(floor, color="#888888", lw=1.0, ls="--", zorder=1)
        ax.annotate(label, (0.35, floor), xycoords=("axes fraction", "data"),
                    ha="left", va="bottom", fontsize=8, color="#666666")

    for stem, (color, label) in ARM_STYLE.items():
        xkey = "step" if stem == "grpo" else "generation"
        pts = curves(stem, xkey)
        xs = [x for x, _ in pts]
        med = [statistics.mean(v) for _, v in pts]  # mean, matching the README and tb3
        lo = [min(v) for _, v in pts]
        hi = [max(v) for _, v in pts]
        ax.plot(xs, med, color=color, lw=1.6, label=label, zorder=3)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)

    ax.set_xlabel("training sample evaluations")
    ax.set_ylabel("held-out reward (2000 puzzles, greedy)")
    ax.set_xlim(0, 500 * EVALS_PER_UNIT)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "f7-e13-heldout.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
