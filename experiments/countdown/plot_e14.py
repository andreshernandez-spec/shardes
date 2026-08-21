#!/usr/bin/env python
"""F8: the robustness curves, from results/e14-a100-2026-08-20. No hand numbers.

    python plot_e14.py       # figures/f8-robustness.png

One panel per algorithm. x = the multiplier applied to the published setting
(log scale), y = final held-out reward at the 150-step horizon, one marker per
seed, median line per dial. The x1 points are E13's committed runs read at the
same horizon. Collapsed runs (final below the arm's decoder floor) are drawn
as red crosses AT their value, which for GRPO at lr x8 is 0.000: the point of
the figure is that the same 64x learning-rate range that ES crosses without a
collapse takes GRPO to zero. The ablation arms (kl_beta 0, clip widened to
never bind) sit at x=1 with their own markers: both trained, which is reported
with the same prominence as the collapses, because an ablation that could not
have come back fine would prove nothing.
"""

from __future__ import annotations

import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
E14 = HERE / "results" / "e14-a100-2026-08-20"
E13 = HERE / "results" / "e13-a100-2026-08-17"
FIGURES = HERE / "figures"
HORIZON = 150
FLOOR = {"es": 0.054, "grpo": 0.037}
INK, MUTED = "#0b0b0b", "#52514e"


def final_at_horizon(path: pathlib.Path) -> float:
    rows = [json.loads(l) for l in path.open()]
    key = "generation" if "generation" in rows[0] else "step"
    return [r["eval_reward"] for r in rows if r[key] <= HORIZON][-1]


def seeds(cell: str) -> list[float]:
    return [final_at_horizon(E14 / f"{cell}-s{s}" / "eval.jsonl") for s in (0, 1, 2)]


def x1(stem: str) -> list[float]:
    return [final_at_horizon(E13 / f"{stem}-s{s}-eval.jsonl") for s in (0, 1, 2)]


def draw_dial(ax, xs, series, color, label, floor):
    med = [statistics.median(v) for v in series]
    ax.plot(xs, med, color=color, lw=1.8, marker="o", ms=5, label=label, zorder=3)
    for x, vals in zip(xs, series):
        for v in vals:
            if v < floor:
                ax.plot([x], [v], marker="x", ms=9, mew=2.2, color="#d62728",
                        zorder=4)
            else:
                ax.plot([x], [v], marker="o", ms=3.5, color=color, alpha=0.55,
                        zorder=2)


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    fig, (ax_es, ax_g) = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)

    draw_dial(ax_es, [1 / 8, 1, 8],
              [seeds("es-lr-eighth"), x1("es-mirrored-lr1"), seeds("es-lr-8x")],
              "#2171b5", "learning rate", FLOOR["es"])
    draw_dial(ax_es, [1 / 4, 1, 4],
              [seeds("es-sigma-quarter"), x1("es-mirrored-lr1"),
               seeds("es-sigma-4x")],
              "#6baed6", "noise scale $\\sigma$", FLOOR["es"])
    ax_es.axhline(FLOOR["es"], color="#888888", lw=1.0, ls="--", zorder=1)
    ax_es.set_title("ES (mirrored rank 1)", color=INK, fontsize=10, loc="left")

    draw_dial(ax_g, [1 / 8, 1, 8],
              [seeds("grpo-lr-eighth"), x1("grpo"), seeds("grpo-lr-8x")],
              "#d62728", "learning rate", FLOOR["grpo"])
    for cell, marker, label in (("grpo-beta-0", "s", "KL anchor removed"),
                                ("grpo-clip-off", "D", "clip never binds")):
        vals = seeds(cell)
        ax_g.plot([1] * 3, vals, marker=marker, ms=6, lw=0, color="#7f4fc9",
                  alpha=0.8, label=label)
    ax_g.axhline(FLOOR["grpo"], color="#888888", lw=1.0, ls="--", zorder=1)
    ax_g.set_title("GRPO (critic-free RLVR)", color=INK, fontsize=10, loc="left")

    for ax in (ax_es, ax_g):
        ax.set_xscale("log")
        ax.set_xticks([1 / 8, 1 / 4, 1, 4, 8])
        ax.set_xticklabels(["1/8", "1/4", "1", "4", "8"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("multiplier on the published setting")
        ax.legend(frameon=False, fontsize=8, loc="center left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.01, 0.20)
    ax_es.set_ylabel(f"held-out reward at step {HORIZON} (2000 puzzles, greedy)")
    ax_es.annotate("base model", (0.98, FLOOR["es"]),
                   xycoords=("axes fraction", "data"), ha="right", va="bottom",
                   fontsize=7, color="#666666")

    fig.tight_layout()
    out = FIGURES / "f8-robustness.png"
    fig.savefig(out, dpi=200)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
