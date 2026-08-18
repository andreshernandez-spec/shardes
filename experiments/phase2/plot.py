#!/usr/bin/env python
"""M1, M2 and M3 as figures.

    python plot.py                                          # the current set, into figures/
    python plot.py --results results-rehearsal --out figures-rehearsal

`docs/03` makes producing the figures part of the dress rehearsal, not of the rented
session: "the plotting script runs end-to-end on the rehearsal data and produces the final
figures". So this has to survive data that is deliberately too small, and say which figures
it could not draw rather than drawing a misleading one.

**Anything produced from simulated CPU devices is watermarked.** Simulated devices share a
memory space and never actually communicate, so a wall-clock scaling curve from them is not
a scaling curve. The watermark is in the image rather than the caption because images get
pasted into slides without their captions.

Colour follows the entity, never its rank: each strategy keeps its hue as configurations are
filtered in and out. Five hues, assigned in fixed order rather than cycled, so adding a
strategy never repaints the ones already there.
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

#: Keyed by entity so a filtered plot never repaints the survivors. Slot 5 is `mirrored_seed`,
#: added 2026-08-14. A strategy with no entry raises a KeyError at plot time rather than
#: silently recycling a colour, which is the right way for this to fail.
HUES = {"iid_gaussian": "#2a78d6", "seed_regenerated": "#eb6834", "mirrored_lr1": "#1baf7a",
        "lowrank_r1": "#eda100", "mirrored_seed": "#8a5fd0"}
MARKERS = {"A": "o", "B": "s"}
INK, MUTED = "#0b0b0b", "#52514e"

#: Two hues plus a neutral grey midpoint. A rainbow here would imply an ordering the data
#: does not have, and a hue at the midpoint would hide the crossover, which IS the result.
CROSSOVER = LinearSegmentedColormap.from_list(
    "crossover", ["#2a78d6", "#d9d9d6", "#eb6834"]
)


def load(results: list[pathlib.Path]) -> list[dict]:
    """Rows from one or more results directories.

    **Several, because a strategy added later lands in its own directory.** `results-qiu`
    holds `mirrored_seed` and `results-consistent` holds the other four, and a figure that
    showed only one of them would be the omission bug this file has already had three times.
    Copying them into a combined directory works and leaves a third copy of every result on
    disk that goes stale the moment either source changes.

    Rows carry their own `env.commit`, so a caller combining directories from different
    commits still has the provenance per row. Whether combining them is *legitimate* is the
    caller's judgement, not this function's.
    """
    rows = []
    for d in results:
        rows += [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
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
    """M1. Wall-clock per generation over parallel efficiency, one column per (model size, N).

    Two rows rather than two y-axes: they have different units, and a dual-axis chart lets
    the reader infer a crossing that is an artefact of the scale choice.

    **This keyed on `(strategy, how)` alone until 2026-08-10, which is the third time that
    bug has appeared in this file.** The sweep runs four `(d_model, population)` blocks, so
    all four collapsed onto every line and whichever sorted last won each device count. By
    filename order the survivor was `d=512, N=256` at every device count, so the efficiency
    ratios happened to stay within one block and the figure was wrong by omission rather
    than by arithmetic: it showed a quarter of the sweep and was captioned as if it showed
    all of it. Nothing about that was guaranteed. Add a configuration whose name sorts later
    at one device count only, and `T_1` and `T_8` come from different shapes, which makes the
    ratio meaningless while still drawing a perfectly plausible line.

    M2 and M3 had the same defect and say so in their own docstrings. A dict key missing a
    factor of the design produces a plausible line, not an error, which is why it survives
    review three times.
    """
    series: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        c = r["config"]
        if c["mode"] != "strong":
            continue
        series[(c["d_model"], c["population"])][(c["strategy"], c["how"])][c["devices"]] = (
            r["seconds_median"]
        )
    if not series:
        return "no strong-scaling rows"

    cells = sorted(series)
    fig, axes = plt.subplots(2, len(cells), figsize=(3.7 * len(cells), 7.4), squeeze=False)
    for j, cell in enumerate(cells):
        d_model, population = cell
        ax1, ax2 = axes[0][j], axes[1][j]
        top = 1.15
        for (strategy, how), by_d in sorted(series[cell].items()):
            ds = sorted(by_d)
            ts = [by_d[d] for d in ds]
            ax1.plot(ds, ts, marker=MARKERS[how], ms=7, lw=2, color=HUES[strategy])
            # Normalized to the smallest MEASURED device count, not to D=1. On the TPU
            # v5e-8 sweep iid_gaussian/B has no D=1 point (it OOMs a 16 GB chip at
            # d=2048, N=256), and dividing its T_2 by D produced a line pinned at 0.5
            # that read as an efficiency collapse. With d0=1 this is the same formula
            # as before; with d0>1 the series reads 1.0 at its own baseline, which is
            # what "efficiency" can honestly mean for it.
            d0, t0 = ds[0], ts[0]
            eff = [(t0 * d0) / (d * by_d[d]) for d in ds]
            top = max(top, max(eff) * 1.05)
            ax2.plot(ds, eff, marker=MARKERS[how], ms=7,
                     lw=2, color=HUES[strategy], label=f"{strategy} / {how}")

        # No single ideal line on the time panel. With several series it can only be anchored
        # to one of them, and it then reads as a target the others are failing to hit. The
        # efficiency panel carries the ideal reference honestly, at 1.0, for every series.
        ax2.axhline(1.0, ls="--", lw=1.5, color=MUTED,
                    label="ideal" if j == 0 else None)

        ideal_d = sorted({d for by_d in series[cell].values() for d in by_d})
        ax1.set(xscale="log", yscale="log", xlabel="devices")
        # The top is data-driven because TPU superlinear points (compiler gets a better
        # layout at higher D) were being clipped by a fixed 1.15.
        ax2.set(xscale="log", xlabel="devices", ylim=(0, top))
        if j == 0:
            ax1.set_ylabel("seconds / generation")
            ax2.set_ylabel("parallel efficiency  $D_0 T_{D_0}/(D\\,T_D)$")
        for ax in (ax1, ax2):
            ax.set_xticks(ideal_d)
            ax.set_xticklabels([str(d) for d in ideal_d])
            ax.xaxis.set_minor_locator(NullLocator())
            _style(ax)
        ax1.set_title(f"d={d_model}, N={population}", color=INK, loc="left", fontsize=10)

    axes[0][0].set_title(
        f"M1  strong scaling: fixed total population\nd={cells[0][0]}, N={cells[0][1]}",
        color=INK, loc="left", fontsize=10,
    )
    # Built from every panel's handles, not from the last one's. A resumed or partial sweep
    # can leave the rightmost block holding a subset of the strategies, and a legend taken
    # from it silently documents four series while eight are drawn.
    handles: dict = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes[1][-1].legend(handles.values(), handles.keys(), frameon=False, fontsize=8,
                       labelcolor=MUTED, loc="center left", bbox_to_anchor=(1.02, 0.5))

    if simulated(rows):
        watermark(fig)
    fig.tight_layout()
    fig.savefig(out / "m1-strong-scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def weak_scaling(rows, out: pathlib.Path) -> str | None:
    """M2 throughput over M6 memory, one column per (model size, population per device).

    **A weak-scaling series is only a series within one experiment**, and this used to key on
    `(strategy, how)` alone. The sweep runs two model sizes and two per-device populations,
    so four experiments collapsed onto every line and whichever sorted last won each device
    count. That is what produced the "non-monotonic memory" in the first committed figure:
    `iid_gaussian/A` read 1610, 810, 6410, 12810 MiB, which is not a measurement, it is
    `d=512, N=128` and `d=512, N=32` interleaved. Split out, both are clean doublings
    (1610, 3210, 6410, 12810 and 486, 810, 1610, 3210), which is what strategy A should do
    in weak mode: every device regenerates the whole population, so per-device storage
    tracks total N.

    Same bug as the one M3 had. Worth stating twice because it is silent both times: a
    dict key that is missing a factor of the design produces a plausible line, not an error.

    Memory on a log axis because the range is 15 MiB to 50 GB, and the small end is the
    result worth seeing.
    """
    series: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        c = r["config"]
        if c["mode"] != "weak":
            continue
        per_device = c["population"] // c["devices"]
        series[(c["d_model"], per_device)][(c["strategy"], c["how"])][c["devices"]] = (
            c["population"] / r["seconds_median"],
            (r.get("peak_bytes_per_device") or 0) / 2**20,
        )
    if not series:
        return "no weak-scaling rows"

    cells = sorted(series)
    fig, axes = plt.subplots(2, len(cells), figsize=(3.7 * len(cells), 7.4), squeeze=False)
    for j, cell in enumerate(cells):
        d_model, per_device = cell
        ax_t, ax_m = axes[0][j], axes[1][j]
        for (strategy, how), by_d in sorted(series[cell].items()):
            ds = sorted(by_d)
            ax_t.plot(ds, [by_d[d][0] for d in ds], marker=MARKERS[how], ms=7, lw=2,
                      color=HUES[strategy], label=f"{strategy} / {how}")
            mem = [by_d[d][1] for d in ds]
            if any(mem):
                ax_m.plot(ds, mem, marker=MARKERS[how], ms=7, lw=2, color=HUES[strategy])

        ds = sorted({d for by_d in series[cell].values() for d in by_d})
        first = next(iter(series[cell].values()))
        ax_t.plot(ds, [first[min(first)][0] * d for d in ds], ls="--", lw=1.5, color=MUTED,
                  label="ideal (linear)")

        ax_t.set(xscale="log", yscale="log", xlabel="devices",
                 title=f"d={d_model}, N/device={per_device}")
        ax_m.set(xscale="log", yscale="log", xlabel="devices")
        for ax in (ax_t, ax_m):
            ax.set_xticks(ds)
            ax.set_xticklabels([str(d) for d in ds])
            ax.xaxis.set_minor_locator(NullLocator())
            _style(ax)
    axes[0][0].set_ylabel("members / second")
    axes[1][0].set_ylabel("peak MiB / device")

    # Handles come from a throughput panel, which is where the labelled artists are. Asking
    # a memory panel for its own would silently produce an empty legend, and did: this
    # figure once shipped with eight series and no key to any of them. matplotlib says so on
    # stderr and then draws it anyway.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=MUTED,
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("M2  weak scaling: fixed population per device      "
                 "M6  peak memory per device (lower row)",
                 color=INK, x=0.02, ha="left", y=1.0)

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
    # `nargs="+"` makes this a list, so the default has to be one too. It was a bare Path
    # for one commit and `plot.py --out ...` raised "PosixPath object is not iterable".
    #
    # The defaults are the current published set, so a bare `plot.py` reproduces `figures/`
    # rather than overwriting it. They used to be `results/` and `figures/`, which was
    # harmless while `figures/` held the first run and actively wrong once it held the
    # current one: the no-argument form would have redrawn the canonical figures from the
    # oldest, pre-scan-fix data.
    ap.add_argument("--results", type=pathlib.Path, nargs="+",
                    default=[HERE / "results-consistent", HERE / "results-qiu"])
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
