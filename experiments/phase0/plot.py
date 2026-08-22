#!/usr/bin/env python
"""Regenerate figure F5 from results/. No manual steps, no hand-edited numbers.

    python plot.py                      # the paper's F5: sigma 1e-3, centered ranks
    python plot.py --sigma 0.01 --shaping baseline   # any other slice

The DEFAULTS are the paper's slice (sigma = 0.001, centered-rank shaping), the
slice the task prediction and the E15 bridge use; running the bare command
reproduces the committed figure. Changing either flag plots a different slice
and must not be committed over f5-estimator-quality.png.

F5: log-log, x = N/d_samp, y = cos(g_hat, grad). Three panels (full rank,
rank 1, rank 4), one curve per scheme with an IQR band, and a vertical line
at N/d_samp = 1. The result records retain the historical field name d_eff;
the paper calls this quantity the sampling dimension.
"""

from __future__ import annotations

import argparse
import json
import math
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


# The shaping axis is conditional on the scheme (docs/01 C0.5), so **no single shaping mode
# exists across all four schemes**: iid schemes carry {centered, centered_ranks} and mirrored
# schemes carry {none, centered_ranks}. F5 wants one curve per scheme on one panel, so it has
# to select by *role* rather than by literal mode.
#
# `baseline` is each scheme's unbiased, variance-reduced arm: `centered` on the iid side, and
# `none` under mirroring, where the pair already cancels f_bar so `none` is centred by
# construction and `centered` would over-correct. Same role, different name, which is exactly
# what the conditional axis is saying.
#
# This role-based baseline is available for diagnostics. The paper defaults to
# `centered_ranks`, matching the update used by E13, E15, and E16.
BASELINE = {"iid": "centered", "mirrored": "none"}


def scheme_side(scheme: str) -> str:
    """Mirror of run.py's `shaping_for` predicate. Duplicated rather than imported, because
    importing the driver pulls in jax for a plotting script; a drift guard in
    tests/test_phase0_driver.py asserts the two agree."""
    return "mirrored" if "mirrored" in scheme else "iid"


def wanted_shaping(scheme: str, shaping: str) -> str:
    return BASELINE[scheme_side(scheme)] if shaping == "baseline" else shaping


def load(sigma: float | None, shaping: str | None) -> list[dict]:
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        rec = json.loads(path.read_text())
        cfg = rec["config"]
        if sigma is not None and abs(cfg["sigma"] - sigma) > 1e-12:
            continue
        if shaping is not None and cfg["shaping"] != wanted_shaping(cfg["scheme"], shaping):
            continue
        records.append(rec)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults are the paper's F5 slice; see the docstring.
    ap.add_argument("--sigma", type=float, default=0.001)
    ap.add_argument("--shaping", default="centered_ranks",
                    help="a literal mode, or 'baseline' for each scheme's unbiased arm "
                         "(centered on the iid side, none under mirroring). See BASELINE.")
    # docs/01 C0.5 specifies y = 1 - cos, which is the right transform when cosine
    # approaches 1: it turns "almost perfect" into a readable decade. Measured, cosine on
    # the transformer block spans about 0.008 to 0.1, so 1 - cos lands in [0.9, 1.0] and a
    # log axis spends its whole range on the third decimal. Plotting cosine directly gives
    # more than a decade of legible range over the same data. Both are kept: switch back
    # with --y one-minus-cos if a future model gets close enough to 1 for it to mean
    # something.
    ap.add_argument("--y", choices=("cos", "one-minus-cos"), default="cos")
    ap.add_argument("--out", type=Path, default=FIGURES / "f5-estimator-quality.png")
    args = ap.parse_args(argv)

    records = load(args.sigma, args.shaping)
    if not records:
        print(f"no results in {RESULTS} for sigma={args.sigma} shaping={args.shaping}")
        return 1

    synthetic = any(r.get("SYNTHETIC") for r in records)
    truncated = sum(bool(r.get("truncated")) for r in records)

    # (rank, scheme) -> list of (N/d_samp, median, q1, q3)
    #
    # The legacy d_eff field is read from the record, not recomputed. The driver knows the model's actual
    # shape; reconstructing it here is how the x-axis drifts the first time the block
    # changes. Note the panels' sampling dimensions differ by perturbation rank: see
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
            # Log-log slope on the medians; the F5 caption quotes the range.
            lx = [math.log(v) for v in x]
            ly = [math.log(m) for _, m, _, _ in points]
            mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
            slope = (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
                     / sum((a - mx) ** 2 for a in lx))
            print(f"slope rank={rank} scheme={scheme}: {slope:.3f}")
            f = (lambda v: 1 - v) if args.y == "one-minus-cos" else (lambda v: v)
            ax.plot(x, [f(m) for _, m, _, _ in points], marker=marker, color=colour,
                    label=label, lw=1.6, ms=5)
            # Bands are IQR over replicates, not a standard error: R = 30 gives wide
            # bars at large N and hiding that would misrepresent the evidence.
            lo = [f(q) for *_, q in points]
            hi = [f(q) for _, _, q, _ in points]
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15)

        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N / d_{\mathrm{samp}}$")
        ax.set_title("full rank" if rank == FULL_RANK else f"rank {rank}")
        ax.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"$1 - \cos(\hat{g}, \nabla f)$" if args.y == "one-minus-cos"
                       else r"$\cos(\hat{g}, \nabla f)$")
    axes[-1].legend(frameon=False, fontsize=9)

    caption = f"E1  sigma={args.sigma}  shaping={args.shaping}"
    if args.shaping == "baseline":
        modes = sorted({wanted_shaping(r["config"]["scheme"], "baseline") for r in records})
        caption += f" ({'/'.join(modes)} per scheme)"
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
