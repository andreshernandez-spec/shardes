#!/usr/bin/env python
"""M1, M2 and M3 as figures.

    python plot.py --results results-rehearsal --out figures/

`docs/03` makes producing the figures part of the dress rehearsal, not of the rented
session: "the plotting script runs end-to-end on the rehearsal data and produces the final
figures". So this has to survive data that is deliberately too small, and say which figures
it could not draw rather than drawing a misleading one.

**Anything produced from simulated CPU devices is watermarked.** Simulated devices share a
memory space and never actually communicate, so a wall-clock scaling curve from them is not
a scaling curve. The watermark is in the image rather than the caption because images get
pasted into slides without their captions.

Colour follows the entity, never its rank: each strategy keeps its hue as configurations are
filtered in and out. The three hues are slots 1-3 of the reference categorical order, used in
fixed order rather than cycled.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

#: Slots 1-3 of the reference categorical order, in order. Keyed by entity so a filtered
#: plot never repaints the survivors.
HUES = {"iid_gaussian": "#2a78d6", "seed_regenerated": "#eb6834", "mirrored_lr1": "#1baf7a",
        "lowrank_r1": "#eda100"}
MARKERS = {"A": "o", "B": "s"}
INK, MUTED = "#0b0b0b", "#52514e"

#: Two hues plus a neutral grey midpoint. A rainbow here would imply an ordering the data
#: does not have, and a hue at the midpoint would hide the crossover, which IS the result.
CROSSOVER = LinearSegmentedColormap.from_list(
    "crossover", ["#2a78d6", "#d9d9d6", "#eb6834"]
)


def load(results: pathlib.Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    return [r for r in rows if "error" not in r]


def simulated(rows: list[dict]) -> bool:
    return any(r.get("env", {}).get("device_platform") == "cpu" for r in rows)


def watermark(fig, text="REHEARSAL - simulated devices, not a scaling measurement"):
    fig.text(0.5, 0.5, text, fontsize=22, color="#e34948", alpha=0.25,
             ha="center", va="center", rotation=18, zorder=10)


def _style(ax):
    ax.grid(True, color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def strong_scaling(rows, out: pathlib.Path) -> str | None:
    """M1. Wall-clock per generation and parallel efficiency, side by side.

    Two panels rather than two y-axes: they have different units, and a dual-axis chart
    lets the reader infer a crossing that is an artefact of the scale choice.
    """
    series = collections.defaultdict(dict)
    for r in rows:
        c = r["config"]
        if c["mode"] != "strong":
            continue
        series[(c["strategy"], c["how"])][c["devices"]] = r["seconds_median"]
    if not series:
        return "no strong-scaling rows"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (strategy, how), by_d in sorted(series.items()):
        ds = sorted(by_d)
        ts = [by_d[d] for d in ds]
        ax1.plot(ds, ts, marker=MARKERS[how], ms=8, lw=2, color=HUES[strategy])
        ax2.plot(ds, [ts[0] / (d * by_d[d]) for d in ds], marker=MARKERS[how], ms=8, lw=2,
                 color=HUES[strategy], label=f"{strategy} / {how}")

    ideal_d = sorted({d for by_d in series.values() for d in by_d})
    # No single ideal line on the left panel. With several series it can only be anchored to
    # one of them, and it then reads as a target the others are failing to hit. The
    # efficiency panel carries the ideal reference honestly, at 1.0, for every series at once.
    ax2.axhline(1.0, ls="--", lw=1.5, color=MUTED, label="ideal")

    ax1.set(xscale="log", yscale="log", xlabel="devices", ylabel="seconds / generation")
    ax2.set(xscale="log", xlabel="devices", ylabel="parallel efficiency  $T_1/(D\\,T_D)$",
            ylim=(0, 1.15))
    for ax in (ax1, ax2):
        ax.set_xticks(ideal_d)
        ax.set_xticklabels([str(d) for d in ideal_d])
        ax.xaxis.set_minor_locator(NullLocator())
        _style(ax)
    ax1.set_title("M1  strong scaling: fixed total population", color=INK, loc="left")
    ax2.set_title("parallel efficiency", color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8, labelcolor=MUTED,
               loc="center left", bbox_to_anchor=(1.02, 0.5))

    if simulated(rows):
        watermark(fig)
    fig.tight_layout()
    fig.savefig(out / "m1-strong-scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def weak_scaling(rows, out: pathlib.Path) -> str | None:
    """M2. Throughput, and per-device memory beside it."""
    thr = collections.defaultdict(dict)
    mem = collections.defaultdict(dict)
    for r in rows:
        c = r["config"]
        if c["mode"] != "weak":
            continue
        thr[(c["strategy"], c["how"])][c["devices"]] = c["population"] / r["seconds_median"]
        if r.get("peak_bytes_per_device"):
            mem[(c["strategy"], c["how"])][c["devices"]] = r["peak_bytes_per_device"] / 2**20
    if not thr:
        return "no weak-scaling rows"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (strategy, how), by_d in sorted(thr.items()):
        ds = sorted(by_d)
        ax1.plot(ds, [by_d[d] for d in ds], marker=MARKERS[how], ms=8, lw=2,
                 color=HUES[strategy], label=f"{strategy} / {how}")
    for (strategy, how), by_d in sorted(mem.items()):
        ds = sorted(by_d)
        ax2.plot(ds, [by_d[d] for d in ds], marker=MARKERS[how], ms=8, lw=2,
                 color=HUES[strategy])

    ds = sorted({d for by_d in thr.values() for d in by_d})
    first = next(iter(thr.values()))
    ax1.plot(ds, [first[min(first)] * d for d in ds], ls="--", lw=1.5, color=MUTED,
             label="ideal (linear)")

    ax1.set(xscale="log", yscale="log", xlabel="devices", ylabel="members / second")
    ax2.set(xscale="log", xlabel="devices", ylabel="peak MiB / device")
    for ax in (ax1, ax2):
        ax.set_xticks(ds)
        ax.set_xticklabels([str(d) for d in ds])
        ax.xaxis.set_minor_locator(NullLocator())
        _style(ax)
    ax1.set_title("M2  weak scaling: fixed population per device", color=INK, loc="left")
    ax2.set_title("M6  peak memory per device", color=INK, loc="left")
    # Handles come from ax1, which is where the labelled artists are. Asking ax2 for its own
    # would silently produce an empty legend, and did: the memory panel plots the same
    # entities without labels, so this figure shipped with eight series and no key to them.
    # matplotlib says so on stderr and then draws the figure anyway.
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, frameon=False, fontsize=8, labelcolor=MUTED,
               loc="center left", bbox_to_anchor=(1.02, 0.5))

    if simulated(rows):
        watermark(fig)
    fig.tight_layout()
    fig.savefig(out / "m2-weak-scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def crossover(rows, out: pathlib.Path) -> str | None:
    """M3. Where strategy B overtakes A, in (N, d).

    Diverging, because the quantity has a meaningful zero: log(t_B / t_A) is negative where
    B wins, positive where A wins, and the contour at 0 is the crossover. That contour is
    the result, so it is drawn explicitly rather than left to the colour scale.

    **One panel per perturbation strategy, and that is not decoration.** The grid used to be
    keyed on `(N, d, how)` with no strategy in the key, so all four strategies wrote to the
    same cell and the last one sorted won: every published cell was `seed_regenerated` and
    the other three were discarded silently. It mattered. Aggregated that way the figure said
    "B wins everywhere"; per strategy, B wins 10 of 16 cells on the 8x A100 sweep, so A wins
    a quarter of them and the headline was an artifact of a dict key.

    A shared colour scale across panels, because the comparison between strategies is the
    point and per-panel scaling would make a 0.003 difference look like a 0.05 one.
    """
    at_max_d = max(r["config"]["devices"] for r in rows)
    grid = {}
    for r in rows:
        c = r["config"]
        if c["mode"] != "strong" or c["devices"] != at_max_d:
            continue
        grid[(c["strategy"], c["population"], c["d_model"], c["how"])] = r["seconds_median"]

    pops = sorted({k[1] for k in grid})
    dims = sorted({k[2] for k in grid})
    strategies = sorted({k[0] for k in grid})
    if len(pops) < 2 or len(dims) < 2:
        return (f"needs a (N, d) grid; have {len(pops)} population(s) x {len(dims)} "
                "model size(s). The rehearsal config is deliberately one point.")

    panels = {}
    for s in strategies:
        z = np.full((len(dims), len(pops)), np.nan)
        for i, d in enumerate(dims):
            for j, n in enumerate(pops):
                a, b = grid.get((s, n, d, "A")), grid.get((s, n, d, "B"))
                if a and b:
                    z[i, j] = np.log10(b / a)
        panels[s] = z

    everything = np.concatenate([z.ravel() for z in panels.values()])
    lim = float(np.nanmax(np.abs(everything))) or 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(1, len(strategies), figsize=(3.6 * len(strategies), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, strategies):
        z = panels[s]
        im = ax.pcolormesh(pops, dims, z, cmap=CROSSOVER, norm=norm, shading="nearest")
        # **Cell values, not a contour.** `docs/03` asks for a crossover contour, and on this
        # grid that would be a lie: two model sizes by three populations with holes in it,
        # so any zero contour is interpolation between four filled cells and moves wherever
        # the interpolator likes. The numbers are the result at this resolution. Draw the
        # contour when the grid is dense enough to carry one.
        for i, d in enumerate(dims):
            for j, n in enumerate(pops):
                if not np.isnan(z[i, j]):
                    ax.text(n, d, f"{z[i, j]:+.3f}", ha="center", va="center",
                            fontsize=8, color=INK)
        if np.nanmin(z) < 0 < np.nanmax(z):
            note = ""
        else:
            note = f"\n{'B' if np.nanmax(z) < 0 else 'A'} wins everywhere"
        ax.set(xscale="log", yscale="log", xlabel="population N", title=f"{s}{note}")
        # `shading="nearest"` centres each cell on its coordinate, so the outer half of the
        # edge cells sits outside the data range and the labels in them get clipped.
        ax.set_ylim(dims[0] / 1.6, dims[-1] * 1.6)
        ax.set_xlim(pops[0] / 1.6, pops[-1] * 1.6)
        ax.set_xticks(pops)
        ax.set_xticklabels([str(n) for n in pops])
        ax.set_yticks(dims)
        ax.set_yticklabels([str(d) for d in dims])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        _style(ax)
    axes[0].set_ylabel("model dimension d")

    fig.colorbar(im, ax=axes.tolist(),
                 label="$\\log_{10}(t_B / t_A)$    <0 B wins,  >0 A wins")
    fig.suptitle(f"M3  contraction crossover at D={at_max_d}", color=INK, x=0.02, ha="left",
                 y=1.04)
    if simulated(rows):
        watermark(fig)
    fig.savefig(out / "m3-crossover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "figures")
    args = ap.parse_args(argv)

    rows = load(args.results)
    if not rows:
        print(f"no usable results in {args.results}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    if simulated(rows):
        print("NOTE: results came from simulated CPU devices. Figures are watermarked and "
              "no timing from them is a scaling measurement.")

    for name, fn in (("M1", strong_scaling), ("M2", weak_scaling), ("M3", crossover)):
        why = fn(rows, args.out)
        print(f"  {name}: {'SKIPPED - ' + why if why else 'written'}")
    print(f"\n{len(rows)} results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
