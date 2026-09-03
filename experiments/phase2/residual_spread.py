#!/usr/bin/env python
"""Is a cell's cost-model residual bigger than the noise on the measurement?

    python residual_spread.py                          # the committed A100 sweep
    python residual_spread.py --sweeps results-iid512-recheck

`timemodel.py` gives each cell a residual, `delta_measured - delta_predicted`.
Whether a residual means anything depends on how well `delta_measured` is itself
pinned down, and the sweep times each placement only a handful of times. This
puts an error bar on the difference of medians so the two can be compared.

The bar is a bootstrap: resample each placement's repeats with replacement,
recompute `median(A) - median(B)`, and take the standard deviation over many
resamples. Nothing here assumes a distribution, which matters because five
timings are far too few to check one. It is deliberately the noise on the
*difference*, since that is the quantity the residual is a residual of.

The A100 sweep's README quoted a spread for two cells before this script
existed. Those figures do not reproduce from the committed records under any
statistic tried, so the numbers this prints replace them.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import timemodel  # noqa: E402


def repeats(dirs, devices: int) -> dict:
    """(strategy, d, N) -> {"A": [seconds], "B": [seconds]}, strong scaling."""
    cells = collections.defaultdict(dict)
    for name in dirs:
        for path in (HERE / name).glob("*.json"):
            rec = json.loads(path.read_text())
            cfg = rec.get("config", {})
            if cfg.get("mode") != "strong" or cfg.get("devices") != devices:
                continue
            if "seconds_all" not in rec:
                continue
            key = (cfg["strategy"], cfg["d_model"], cfg["population"])
            cells[key][cfg["how"]] = rec["seconds_all"]
    return {k: v for k, v in cells.items() if "A" in v and "B" in v}


def noise(a: list, b: list, draws: int, rng) -> tuple[float, float]:
    """Bootstrap sd and 95% half-width of median(a) - median(b), in seconds."""
    ra = rng.choice(np.asarray(a), size=(draws, len(a)), replace=True)
    rb = rng.choice(np.asarray(b), size=(draws, len(b)), replace=True)
    d = np.median(ra, axis=1) - np.median(rb, axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.std(d)), float((hi - lo) / 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="*", default=None,
                    help="sweep directories; default is the A100 platform's")
    ap.add_argument("--devices", type=int, default=timemodel.DEVICES)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    platform, spec = next(iter(timemodel.PLATFORMS.items()))
    dirs = args.sweeps or spec["sweeps"]
    rows, _ = timemodel.rows(platform, spec, devices=args.devices)
    by_key = {(r["strategy"], r["d_model"], r["population"]): r for r in rows}
    reps = repeats(dirs, args.devices)
    rng = np.random.default_rng(args.seed)

    print(f"## {platform}, D={args.devices}, from {', '.join(dirs)}")
    print(f"bootstrap of median(A) - median(B), {args.draws} draws, seed {args.seed}\n")
    print("| strategy | d | N | repeats | residual ms | noise sd ms | 95% +- ms | "
          "residual / sd |")
    print("|---|---|---|---|---|---|---|---|")
    loud = []
    for key in sorted(reps):
        row = by_key.get(key)
        if row is None or row.get("delta_predicted") is None:
            continue
        a, b = reps[key]["A"], reps[key]["B"]
        sd, half = noise(a, b, args.draws, rng)
        resid = row["delta_measured"] - row["delta_predicted"]
        ratio = abs(resid) / sd if sd > 0 else float("inf")
        strategy, d, n = key
        print(f"| {strategy} | {d} | {n} | {len(a)}/{len(b)} | {resid * 1e3:+.3f} | "
              f"{sd * 1e3:.3f} | {half * 1e3:.3f} | {ratio:.1f} |")
        if ratio > 2:
            loud.append((ratio, strategy, d, n, resid * 1e3, sd * 1e3))

    print()
    if not loud:
        print("No cell's residual exceeds twice the noise on its own measurement.")
        return 0
    print(f"{len(loud)} of {len(reps)} cells exceed twice the noise on their own "
          "measurement:")
    for ratio, strategy, d, n, resid, sd in sorted(loud, reverse=True):
        print(f"  {strategy} d={d} N={n}: {resid:+.3f} ms against {sd:.3f} ms "
              f"({ratio:.1f}x)")
    med = statistics.median(r[0] for r in loud)
    print(f"  median ratio among them {med:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
