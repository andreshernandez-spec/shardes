#!/usr/bin/env python
"""F9: the real-model contraction crossover as a figure, from results-e17b.

    python plot_e17.py            # writes figures/f9-e17-crossover.png

One panel per perturbation arm, x = device count, y = log10(t_B / t_A) of
median generation time, one line per population. Negative: the model-sized
all-reduce placement (B) wins. A cell where either placement ran out of
memory is drawn as a hollow marker at y=0 with no line through it; a cell
with no record at all (killed in compilation) is absent. Falls back to
results-e17 when results-e17b does not exist yet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-e17b" if (HERE / "results-e17b").exists() else HERE / "results-e17"
FIGURES = HERE / "figures"
ARMS = [("mirrored_seed", "full rank (seed)"), ("mirrored_lr1", "rank 1"),
        ("mirrored_lr4", "rank 4"), ("mirrored_lr16", "rank 16")]
DEVICES = (1, 2, 4, 8)
SHADES = {32: "#08306b", 64: "#2171b5", 128: "#4292c6", 240: "#9ecae1"}


def cell(strategy, how, n, d):
    f = RESULTS / f"s={strategy}__how={how}__N={n}__D={d}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("seconds_median", "oom")


def main() -> None:
    pops = sorted({int(f.name.split("N=")[1].split("__")[0]) for f in RESULTS.glob("s=*.json")})
    arms = [(s, l) for s, l in ARMS if any(RESULTS.glob(f"s={s}__*.json"))]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.3 * len(arms), 3.6), sharey=True, squeeze=False)
    for ax, (strategy, label) in zip(axes[0], arms):
        for n in pops:
            xs, ys, ooms = [], [], []
            for d in DEVICES:
                a, b = cell(strategy, "A", n, d), cell(strategy, "B", n, d)
                if a is None or b is None:
                    continue
                if a == "oom" or b == "oom":
                    ooms.append(d)
                    continue
                xs.append(d)
                ys.append(math.log10(b / a))
            color = SHADES.get(n, "#444444")
            if xs:
                ax.plot(xs, ys, marker="o", ms=5, lw=1.6, color=color, label=f"N={n}")
            if ooms:
                ax.scatter(ooms, [0.0] * len(ooms), facecolors="none", edgecolors=color,
                           s=36, lw=1.2, zorder=3)
        ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEVICES)
        ax.set_xticklabels([str(d) for d in DEVICES])
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xlabel("devices")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(r"$\log_{10}(t_B / t_A)$   (negative: all-reduce wins)")
    # One legend for the figure: every population that drew a line anywhere,
    # plus the hollow marker, since a panel may have no line for a population.
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    handles["OOM (either placement)"] = plt.Line2D(
        [], [], marker="o", ls="none", markerfacecolor="none", color="#444444")
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8,
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "f9-e17-crossover.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
