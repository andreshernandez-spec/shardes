#!/usr/bin/env python
"""F4: where the low-rank rewrite pays, per platform.

    python plot_cost.py                                 # GPU panel row from results-cost
    python plot_cost.py --results results-cost results-cost-tpu-v5e8   # both rows

One row of panels per platform, one panel per strategy, each a heatmap over
(N, d) of log10(t_strategy / t_iid_gaussian) at the same shape and dtype. The
diverging scale is M3's: blue where the strategy beats the dense baseline,
orange where the baseline wins, neutral grey at parity, because the quantity
has a meaningful zero and the zero is the claim.

**Cells where the dense baseline is infeasible are drawn as wins, not holes.**
On the A100 the baseline OOMs over half the grid; a ratio needs both sides, so
those cells cannot carry a number, but leaving them blank would erase the
strongest form of C4's claim (the rewrite pays by feasibility before it pays by
throughput). They get the full win colour and the label "dense OOM". Cells
where the strategy itself is undersized say OOM; cells where both are say so.

Annotations are multipliers ("0.13x"), not log values, because a reader
checking a cell against the table in the paper should not need to exponentiate.
The colour is still log-scaled so 4x-faster and 4x-slower sit symmetrically.

Default dtype is bfloat16: E8 claims cost at the precision a practitioner runs,
and the tensor-core/MXU story C4 is about is bf16-shaped. --dtype float32
draws the other surface from the same records.

Draft for Andres: the two decisions worth a second opinion are (1) ratio vs
absolute per-member time (ratio was chosen because C4 is a claim about a
comparison, and the absolute surfaces are recoverable from the JSONs), and
(2) bf16 as the headline dtype with f32 relegated to a flag.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
INK, MUTED = "#0b0b0b", "#52514e"
OOM = "oom"  # sentinel: the cell was visited and recorded undersized

#: Same poles as plot.py's CROSSOVER and the same reading: blue = the named thing
#: wins, orange = the baseline wins, grey = parity.
CMAP = LinearSegmentedColormap.from_list("pays", ["#2a78d6", "#d9d9d6", "#eb6834"])

#: Panel order tells the story left to right: Qiu's storage fix first (it loses
#: throughput at D=1), then the mirrored family EGGROLL motivates.
STRATEGIES = ["seed_regenerated", "mirrored_seed", "mirrored_lr1", "mirrored_lr4",
              "mirrored_lr16"]
BASELINE = "iid_gaussian"


def load(dirs: list[pathlib.Path]) -> dict:
    """{platform: {(d, N, strategy, dtype): seconds | OOM}}, one platform per dir.

    Undersized records are kept as the OOM sentinel rather than dropped: the
    feasibility boundary is half of what this figure shows. A directory mixing
    two device kinds raises, because a surface stitched from two machines is
    not a surface.
    """
    out: dict = {}
    for d in dirs:
        rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        kinds = {r["env"]["device_kind"] for r in rows}
        if len(kinds) != 1:
            raise SystemExit(f"{d} mixes device kinds: {sorted(kinds)}")
        cells = {}
        for r in rows:
            c = r["config"]
            key = (c["d_model"], c["population"], c["strategy"], c["dtype"])
            cells[key] = OOM if r.get("undersized") else r["seconds_median"]
        out[kinds.pop()] = cells
    return out


def panel(ax, cells: dict, strategy: str, dtype: str, dims, pops, norm) -> None:
    z = np.full((len(dims), len(pops)), np.nan)
    labels = np.full((len(dims), len(pops)), "", dtype=object)
    for i, d in enumerate(dims):
        for j, n in enumerate(pops):
            t = cells.get((d, n, strategy, dtype))
            base = cells.get((d, n, BASELINE, dtype))
            if t is None or base is None:
                continue
            if t is not OOM and base is not OOM:
                z[i, j] = np.log10(t / base)
                labels[i, j] = f"{t / base:.2f}x"
            elif t is not OOM:  # the baseline is the one that does not fit
                z[i, j], labels[i, j] = norm.vmin, "dense\nOOM"
            elif base is not OOM:
                z[i, j], labels[i, j] = norm.vmax, "OOM"
            else:
                labels[i, j] = "both\nOOM"  # stays NaN: neither side has a time
    ax.pcolormesh(pops, dims, np.ma.masked_invalid(z), cmap=CMAP, norm=norm,
                  shading="nearest")
    for i, d in enumerate(dims):
        for j, n in enumerate(pops):
            if labels[i, j]:
                ax.text(n, d, labels[i, j], ha="center", va="center", fontsize=7,
                        color=INK)
    ax.set(xscale="log", yscale="log", xlabel="population N")
    ax.set_xlim(pops[0] / 1.6, pops[-1] * 1.6)
    ax.set_ylim(dims[0] / 1.6, dims[-1] * 1.6)
    ax.set_xticks(pops)
    # 16384 and its neighbour collide as full digits at this panel width.
    ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in pops],
                       fontsize=8)
    ax.set_yticks(dims)
    ax.set_yticklabels([str(d) for d in dims], fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, nargs="+",
                    default=[HERE / "results-cost"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "figures")
    args = ap.parse_args(argv)

    platforms = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    dims = sorted({k[0] for cells in platforms.values() for k in cells})
    pops = sorted({k[1] for cells in platforms.values() for k in cells})

    # One shared symmetric scale across every panel and platform, so the same
    # shade means the same ratio everywhere; the comparison IS the figure.
    finite = [np.log10(t / cells.get((k[0], k[1], BASELINE, k[3])))
              for cells in platforms.values() for k, t in cells.items()
              if k[2] != BASELINE and k[3] == args.dtype and t is not OOM
              and cells.get((k[0], k[1], BASELINE, k[3])) not in (None, OOM)]
    if not finite:
        print(f"no comparable ({args.dtype}) cells in {list(args.results)}")
        return 1
    lim = float(np.max(np.abs(finite)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    nrows = len(platforms)
    fig, axes = plt.subplots(nrows, len(STRATEGIES),
                             figsize=(2.9 * len(STRATEGIES), 3.4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for i, (kind, cells) in enumerate(sorted(platforms.items())):
        for j, s in enumerate(STRATEGIES):
            panel(axes[i][j], cells, s, args.dtype, dims, pops, norm)
            if i == 0:
                axes[i][j].set_title(s, color=INK, fontsize=10)
            if j == 0:
                axes[i][j].set_ylabel(f"{kind}\nmodel dimension d", color=INK)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                 label="$\\log_{10}(t / t_\\mathrm{dense})$"
                       "    <0 strategy wins,  >0 dense wins")
    fig.suptitle(f"F4  cost vs the materializing dense baseline, {args.dtype}, D=1",
                 color=INK, x=0.02, ha="left", y=1.0)
    out = args.out / f"f4-cost-{args.dtype}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
