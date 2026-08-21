#!/usr/bin/env python
"""F1 and F2, the paper's cross-platform figures, from the committed sweeps.

    python plot_paper.py      # figures/f1-scaling.png, figures/f2-crossover.png

Both figures put the A100 row above the v5e row so the platform comparison is a
vertical glance. Sources: `results-consistent` + `results-qiu` (8x A100) and
`results-tpu-v5e8` (v5e-8), the same directories M1-M3 draw from; nothing here
is a new measurement, only an assembly, so any number can be checked against
the per-platform figures.

F2 keeps one panel per strategy, per platform. M3's docstring records why that
is not decoration: aggregating strategies once turned "B wins 10 of 16 cells"
into "B wins everywhere". Only the four strategies present on BOTH platforms
are drawn (`mirrored_seed` exists only in the GPU sweep, results-qiu); the
shared colour scale spans both platforms so the same shade is the same ratio
everywhere, which is the point of stacking them.

F1 is the opener, so it shows one representative cell per mode rather than
M1/M2's full grid: the largest cell both platforms ran (d=2048: strong
N=256, weak N/device=32). Strong shows seconds per generation, log-log, no
per-panel ideal line (M1's docstring: with several series a time-panel ideal
reads as a target the others miss); weak shows members/second with the dashed
linear ideal the figure table asks for, anchored per series family at its own
D=1 point via the same convention as M2. Full grids stay in the M1/M2 figures
the appendix cites.

Draft for Andres: the F1 cell choice (largest common) and the decision to show
time + throughput rather than efficiency in the opener are the two judgement
calls; both are one-line changes here.
"""

from __future__ import annotations

import collections
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot import CROSSOVER, HUES, INK, MARKERS, MUTED, _style, load  # noqa: E402

PLATFORMS = [
    ("8x A100", [HERE / "results-consistent", HERE / "results-qiu"]),
    ("TPU v5e-8", [HERE / "results-tpu-v5e8"]),
]
#: Only what both platforms ran; mirrored_seed is GPU-only (results-qiu).
STRATEGIES = ["iid_gaussian", "seed_regenerated", "mirrored_lr1", "lowrank_r1"]


def f2(platform_rows: list[tuple[str, list[dict]]], out: pathlib.Path) -> None:
    grids = {}
    for name, rows in platform_rows:
        at_max_d = max(r["config"]["devices"] for r in rows)
        g = {}
        for r in rows:
            c = r["config"]
            if c["mode"] != "strong" or c["devices"] != at_max_d:
                continue
            # run.py's seconds_iqr is the width q3-q1 (cost.py's is the pair);
            # normalize to a (lo, hi) bracket around the median either way.
            m, iqr = r["seconds_median"], r.get("seconds_iqr", 0.0)
            lo, hi = ((iqr[0], iqr[1]) if isinstance(iqr, (list, tuple))
                      else (m - iqr / 2, m + iqr / 2))
            g[(c["strategy"], c["population"], c["d_model"], c["how"])] = (m, (lo, hi))
        grids[name] = (at_max_d, g)

    pops = sorted({k[1] for _, g in grids.values() for k in g})
    dims = sorted({k[2] for _, g in grids.values() for k in g})

    panels = {}
    for name, (_, g) in grids.items():
        for s in STRATEGIES:
            z = np.full((len(dims), len(pops)), np.nan)
            noisy = np.zeros((len(dims), len(pops)), bool)
            for i, d in enumerate(dims):
                for j, n in enumerate(pops):
                    a, b = g.get((s, n, d, "A")), g.get((s, n, d, "B"))
                    if a and b:
                        z[i, j] = np.log10(b[0] / a[0])
                        # Within measurement noise if the B/A ratio interval built
                        # from the two IQRs straddles 1: the sign is then not a
                        # finding, and the cell says so instead of implying one.
                        lo, hi = b[1][0] / a[1][1], b[1][1] / a[1][0]
                        noisy[i, j] = lo <= 1.0 <= hi
            panels[(name, s)] = (z, noisy)

    everything = np.concatenate([z.ravel() for z, _ in panels.values()])
    lim = float(np.nanmax(np.abs(everything))) or 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(len(grids), len(STRATEGIES),
                             figsize=(3.1 * len(STRATEGIES), 3.4 * len(grids)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (name, (at_max_d, _)) in enumerate(grids.items()):
        for j, s in enumerate(STRATEGIES):
            ax = axes[i][j]
            z, noisy = panels[(name, s)]
            im = ax.pcolormesh(pops, dims, z, cmap=CROSSOVER, norm=norm,
                               shading="nearest")
            for ii, d in enumerate(dims):
                for jj, n in enumerate(pops):
                    if not np.isnan(z[ii, jj]):
                        label = f"{z[ii, jj]:+.2f}"
                        if noisy[ii, jj]:
                            label = f"({label})"  # sign within measurement noise
                        ax.text(n, d, label, ha="center", va="center",
                                fontsize=8, color=INK)
            ax.set(xscale="log", yscale="log")
            ax.set_xlim(pops[0] / 1.6, pops[-1] * 1.6)
            ax.set_ylim(dims[0] / 1.6, dims[-1] * 1.6)
            ax.set_xticks(pops)
            ax.set_xticklabels([str(n) for n in pops], fontsize=8)
            ax.set_yticks(dims)
            ax.set_yticklabels([str(d) for d in dims], fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            _style(ax)
            ax.grid(False)
            if i == 0:
                ax.set_title(s, color=INK, fontsize=10)
            if i == len(grids) - 1:
                ax.set_xlabel("population N")
            if j == 0:
                ax.set_ylabel(f"{name} (D={at_max_d})\nmodel dimension d", color=INK)

    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label="$\\log_{10}(t_B / t_A)$    <0 B wins,  >0 A wins")
    fig.suptitle("F2  contraction crossover, by platform", color=INK, x=0.02,
                 ha="left", y=0.98)
    fig.savefig(out / "f2-crossover.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out / "f2-crossover.png")


def f2b(platform_rows: list[tuple[str, list[dict]]], out: pathlib.Path) -> None:
    """The crossover's trajectory in device count, from the same sweeps.

    F2 shows D=8 only; a referee's next question is whether the sign pattern is
    a D=8 artifact. One line per (strategy, d, N) cell, log10(t_B/t_A) against
    D, per platform. The lines answer it from data that already existed: the
    sweeps measured every D in {1,2,4,8}.
    """
    fig, axes = plt.subplots(1, len(platform_rows), figsize=(4.6 * len(platform_rows), 3.6),
                             sharey=True, squeeze=False)
    for j, (name, rows) in enumerate(platform_rows):
        ax = axes[0][j]
        cells: dict = collections.defaultdict(dict)
        for r in rows:
            c = r["config"]
            if c["mode"] != "strong" or c["strategy"] not in STRATEGIES:
                continue
            cells[(c["strategy"], c["d_model"], c["population"])][
                (c["devices"], c["how"])] = r["seconds_median"]
        for (s, d, n), by in sorted(cells.items()):
            ds = sorted({dev for dev, _ in by})
            pts = [(dev, np.log10(by[(dev, "B")] / by[(dev, "A")]))
                   for dev in ds if (dev, "A") in by and (dev, "B") in by and dev > 1]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                        lw=1.3, color=HUES[s], alpha=0.85,
                        label=s if (d, n) == min((d2, n2) for (s2, d2, n2) in cells
                                                 if s2 == s) else None)
        ax.axhline(0.0, color=MUTED, lw=1.0, ls="--")
        ax.set(xscale="log", xlabel="devices D")
        ax.set_xticks([2, 4, 8])
        ax.set_xticklabels(["2", "4", "8"])
        ax.xaxis.set_minor_locator(NullLocator())
        _style(ax)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        if j == 0:
            ax.set_ylabel("$\\log_{10}(t_B / t_A)$")
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED)
    fig.suptitle("F2b  the crossover ratio against device count, one line per "
                 "(strategy, d, N) cell", color=INK, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "f2b-crossover-vs-d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out / "f2b-crossover-vs-d.png")


def f1(platform_rows: list[tuple[str, list[dict]]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, len(platform_rows), figsize=(4.4 * len(platform_rows), 6.8),
                             squeeze=False)
    for j, (name, rows) in enumerate(platform_rows):
        ax_s, ax_w = axes[0][j], axes[1][j]

        strong: dict = collections.defaultdict(dict)
        weak: dict = collections.defaultdict(dict)
        for r in rows:
            c = r["config"]
            if c["strategy"] not in STRATEGIES or c["d_model"] != 2048:
                continue
            if c["mode"] == "strong" and c["population"] == 256:
                strong[(c["strategy"], c["how"])][c["devices"]] = r["seconds_median"]
            if c["mode"] == "weak" and c["population"] // c["devices"] == 32:
                weak[(c["strategy"], c["how"])][c["devices"]] = (
                    c["population"] / r["seconds_median"])

        for (s, how), by_d in sorted(strong.items()):
            ds = sorted(by_d)
            ax_s.plot(ds, [by_d[d] for d in ds], marker=MARKERS[how], ms=6, lw=1.8,
                      color=HUES[s], label=f"{s} / {how}")
        # Parallel efficiency, each series against ITS OWN smallest measured
        # device count: eff(D) = (T(D0) * D0) / (T(D) * D) in throughput terms.
        # The panel used to plot absolute members/second against one ideal
        # line anchored to the alphabetically first series, which made
        # cross-series comparisons against that line meaningless (a faster
        # series crossed it without any superlinear scaling). Efficiency
        # gives every series the same ideal (1.0) whatever its D0, which
        # also handles arms whose D=1 cell is out of memory.
        for (s, how), by_d in sorted(weak.items()):
            ds = sorted(by_d)
            d0 = ds[0]
            ax_w.plot(ds, [(by_d[d] / d) / (by_d[d0] / d0) for d in ds],
                      marker=MARKERS[how], ms=6, lw=1.8, color=HUES[s])
            print(f"  weak {name} {s}/{how}: D0={d0} "
                  + " ".join(f"eff(D={d})={(by_d[d] / d) / (by_d[d0] / d0):.2f}"
                             for d in ds))
        ax_w.axhline(1.0, ls="--", lw=1.4, color=MUTED,
                     label="ideal" if j == 0 else None)

        ticks = sorted({d for by_d in strong.values() for d in by_d})
        for ax, ylab, ylog in ((ax_s, "seconds / generation", True),
                               (ax_w, "weak-scaling efficiency", False)):
            ax.set(xscale="log")
            if ylog:
                ax.set_yscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(d) for d in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            _style(ax)
            if j == 0:
                ax.set_ylabel(ylab)
        ax_w.set_ylim(0.0, 1.25)
        ax_s.set_title(f"{name}\nstrong: d=2048, N=256", color=INK, loc="left",
                       fontsize=10)
        ax_w.set_title("weak: d=2048, N/device=32", color=INK, loc="left", fontsize=10)
        ax_w.set_xlabel("devices")

    handles: dict = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=8,
               labelcolor=MUTED, loc="center left", bbox_to_anchor=(0.99, 0.5))
    fig.suptitle("F1  scaling by platform", color=INK, x=0.02, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "f1-scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out / "f1-scaling.png")


def main() -> int:
    platform_rows = [(name, load(dirs)) for name, dirs in PLATFORMS]
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    f2(platform_rows, out)
    f2b(platform_rows, out)
    f1(platform_rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
