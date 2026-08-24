#!/usr/bin/env python
"""F7c: the E19 positive control beside the E13 arms it tests.

    python plot_e19.py [--e13 results/e13-a100-<date>]

Held-out reward against training sample evaluations for full rank and
rank 1 at N=30 (E13) and N=16 (E19), mean over three seeds with a min-max
band. Units differ per arm and are applied per arm: an E13 update scores
30 x 8 = 240 completions, an E19 update 16 x 8 = 128. Prints, per arm, the
final held-out reward and the first evaluation at or above 0.15 in samples,
which is what the control has to resolve for the E13 null to mean anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
E19 = HERE / "results" / "e19-a100"
ARMS = [  # (results dir, stem, evals per update, color, label)
    (None, "es-mirrored-seed", 240, "#08306b", "full rank, N=30 (E13)"),
    (None, "es-mirrored-lr1", 240, "#2171b5", "rank 1, N=30 (E13)"),
    (E19, "es-n16-seed", 128, "#d94801", "full rank, N=16"),
    (E19, "es-n16-lr1", 128, "#fd8d3c", "rank 1, N=16"),
]


def curves(root: Path, stem: str, unit: int):
    by_x: dict = {}
    for s in (0, 1, 2):
        for row in map(json.loads, (root / f"{stem}-s{s}-eval.jsonl").open()):
            by_x.setdefault(row["generation"] * unit, []).append(row["eval_reward"])
    return sorted(by_x.items())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e13", type=Path, default=HERE / "results" / "e13-a100-2026-08-22-clean")
    args = ap.parse_args(argv)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for root, stem, unit, color, label in ARMS:
        root = root or args.e13
        if not (root / f"{stem}-s0-eval.jsonl").exists():
            print(f"missing {stem} under {root}")
            continue
        pts = curves(root, stem, unit)
        xs = [x for x, _ in pts]
        mean = [statistics.mean(v) for _, v in pts]
        ax.plot(xs, mean, color=color, lw=1.6, label=label, zorder=3)
        ax.fill_between(xs, [min(v) for _, v in pts], [max(v) for _, v in pts],
                        color=color, alpha=0.15, lw=0, zorder=2)
        hit = next((x for x, v in pts if statistics.mean(v) >= 0.15), None)
        print(f"{label:24s} final {mean[-1]:.3f} [{min(pts[-1][1]):.3f}, {max(pts[-1][1]):.3f}]"
              f"  first mean>=0.15 at {hit} samples")
    ax.set_xlabel("training sample evaluations")
    ax.set_ylabel("held-out reward (2000 puzzles, greedy)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "f7c-e19-positive-control.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
