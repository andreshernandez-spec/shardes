#!/usr/bin/env python
"""Regenerate figure F5 from results/. No manual steps, no hand-edited numbers.

    python plot.py                      # figures/f5-estimator-quality.png
    python plot.py --sigma 0.01         # pick the sigma slice
    python plot.py --shaping none

F5: log-log, x = N/d_eff, y = 1 - cos(g_hat, grad). Two panels, full rank and rank 1.
One curve per scheme with an IQR band, and a vertical line at N/d_eff = 1.

The claim the figure supports or kills: curves separate in the rank-1 panel to the right
of the line, and do not separate in the full-rank panel to the left of it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this has to work over ssh and in a notebook driver
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Colour per scheme, fixed so the same scheme is the same colour in every figure and in
# any talk that reuses them.
SCHEME_STYLE = {
    "iid": ("#888888", "o", "i.i.d."),
    "mirrored": ("#1f77b4", "s", "mirrored"),
    "mirrored+orthogonal_hd": ("#d62728", "^", "mirrored + orthogonal HD"),
    "mirrored+sobol": ("#2ca02c", "D", "mirrored + scrambled Sobol"),
}


def rank_key(rank) -> tuple:
    """Ranks are ints or the string "full", so they need an explicit ordering. Full rank
    sorts first, which puts it in the left panel where the G0 claim expects it."""
    return (0, 0) if rank == FULL_RANK else (1, rank)


FULL_RANK = "full"


def load(sigma: float | None, shaping: str | None) -> list[dict]:
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        rec = json.loads(path.read_text())
        cfg = rec["config"]
        if sigma is not None and abs(cfg["sigma"] - sigma) > 1e-12:
            continue
        if shaping is not None and cfg["shaping"] != shaping:
            continue
        records.append(rec)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--shaping", default="none")
    ap.add_argument("--out", type=Path, default=FIGURES / "f5-estimator-quality.png")
    args = ap.parse_args(argv)

    records = load(args.sigma, args.shaping)
    if not records:
        print(f"no results in {RESULTS} for sigma={args.sigma} shaping={args.shaping}")
        return 1

    synthetic = any(r.get("SYNTHETIC") for r in records)
    truncated = sum(bool(r.get("truncated")) for r in records)

    # (rank, scheme) -> list of (N/d_eff, median, q1, q3)
    #
    # d_eff is read from the record, not recomputed. The driver knows the model's actual
    # shape; reconstructing it here is how the x-axis drifts the first time the block
    # changes. Note the two panels' d_eff measure different things on purpose: see
    # src/shardes/dimensions.py, and say so in the caption.
    series: dict = defaultdict(list)
    for rec in records:
        cfg = rec["config"]
        if "d_eff" not in rec:
            print(f"skipping a result with no d_eff (pre-{__file__} format): {cfg}")
            continue
        series[(cfg["rank"], cfg["scheme"])].append(
            (cfg["population"] / rec["d_eff"], rec["cosine_median"],
             rec["cosine_q1"], rec["cosine_q3"])
        )
    if not any(series.values()):
        print("no results carried a d_eff field; re-run the sweep")
        return 1

    ranks = sorted({r for r, _ in series}, key=rank_key)
    fig, axes = plt.subplots(1, len(ranks), figsize=(6 * len(ranks), 5), sharey=True)
    axes = [axes] if len(ranks) == 1 else list(axes)

    for ax, rank in zip(axes, ranks):
        for (r, scheme), points in sorted(series.items(), key=lambda kv: (rank_key(kv[0][0]), kv[0][1])):
            if r != rank:
                continue
            colour, marker, label = SCHEME_STYLE.get(scheme, ("#000000", "x", scheme))
            points.sort()
            x = [ratio for ratio, *_ in points]
            ax.plot(x, [1 - m for _, m, _, _ in points], marker=marker, color=colour,
                    label=label, lw=1.6, ms=5)
            # Bands are IQR over replicates, not a standard error: R = 30 gives wide
            # bars at large N and hiding that would misrepresent the evidence.
            ax.fill_between(x, [1 - q3 for *_, q3 in points],
                            [1 - q1 for _, _, q1, _ in points], color=colour, alpha=0.15)

        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N / d_{\mathrm{eff}}$")
        ax.set_title("full rank" if rank == FULL_RANK else f"rank {rank}")
        ax.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"$1 - \cos(\hat{g}, \nabla f)$")
    axes[-1].legend(frameon=False, fontsize=9)

    caption = f"E1  sigma={args.sigma}  shaping={args.shaping}"
    if truncated:
        caption += f"  ({truncated} config(s) truncated by the wall-clock cap)"
    fig.suptitle(caption, fontsize=10)

    if synthetic:
        # A fake figure that looks real is worse than no figure. Make it impossible to
        # screenshot this into a slide by accident.
        for ax in axes:
            ax.text(0.5, 0.5, "SYNTHETIC\ndry run", transform=ax.transAxes,
                    fontsize=34, color="red", alpha=0.28, ha="center", va="center",
                    rotation=25, zorder=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}" + ("  [SYNTHETIC]" if synthetic else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
