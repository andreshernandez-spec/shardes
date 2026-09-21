#!/usr/bin/env python
"""Read the shaping-barrier records properly, and check them against the ladder.

    python barrier_report.py
    python barrier_report.py --results results-barrier-tpu-v5e8

`barrier.py` times three programs per (N, D): shaping `none`, `centered` and
`centered_ranks`. The first write-up read the `none` row as the fitness gather
and reported it at 2.4-6 us, flat in N up to a 1 MiB payload. That reading is
wrong, and the records themselves say so.

Under `none` the step is

    weights = replicate(f)                  # the gather
    f       = reshard(f + 1e-6 * weights)   # elementwise, then back to P("pop")

so each device's next carry depends only on its own slice of the gathered
array. Nothing downstream ever needs the other devices' slices, and the
compiler is free to drop the gather or leave it off the critical path. The
chain's docstring guards against hoisting; it does not guard against this.
`centered` and `centered_ranks` are different: a mean, a standard deviation or
a rank needs every member, so their gather cannot be skipped.

The test is payload dependence. A real all-gather is latency-bound for small
messages and bandwidth-bound for large ones, so what eight devices add to a
program must grow between N=64 and N=2^18. It does for the two programs that
consume the gathered array and does not for `none`. The growth also matches
the dedicated instrument, `allreduce_ladder.py`, which times the same tiled
all-gather of the same payload on the same chip.

What survives: every statement about the sort, which is read at D=1 where
there is no communication to confuse it, and the in-context closure in
`results-barrier-context`. What does not: "the gather costs 2.4-6 us, flat in
N". The gather costs what the ladder says it costs.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
LADDERS = {"tpu": "results-ladder/ladder-tpu-v5-lite-D8.json",
           "gpu": "results-ladder/ladder-nvidia-a100-sxm4-80gb-D8.json"}


def load(results: pathlib.Path) -> tuple[dict, str]:
    cells = collections.defaultdict(dict)
    platform = "tpu"
    for path in results.glob("*.json"):
        rec = json.loads(path.read_text())
        if "seconds_median" not in rec:
            continue
        cfg = rec["config"]
        cells[(cfg["population"], cfg["devices"])][cfg["shaping"]] = rec["seconds_median"]
        platform = rec.get("env", {}).get("device_platform", platform)
    return cells, platform


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results-barrier-tpu-v5e8")
    ap.add_argument("--devices", type=int, default=8)
    args = ap.parse_args(argv)

    cells, platform = load(HERE / args.results)
    d = args.devices
    pops = sorted({n for n, dev in cells if dev == d and (n, 1) in cells})
    ladder = {}
    ladder_file = HERE / LADDERS.get(platform, LADDERS["tpu"])
    if ladder_file.exists():
        ladder = {int(k): v["step_seconds"]
                  for k, v in json.loads(ladder_file.read_text())["allgather"].items()}

    def us(x):
        return f"{x * 1e6:.1f}"

    print(f"## {args.results}: what D={d} adds to each program over D=1, microseconds\n")
    print("| N | payload | none | centered | centered_ranks | ladder all-gather step |")
    print("|---|---|---|---|---|---|")
    added = collections.defaultdict(dict)
    for n in pops:
        one, many = cells[(n, 1)], cells[(n, d)]
        for s in ("none", "centered", "centered_ranks"):
            added[s][n] = many[s] - one[s]
        print(f"| {n} | {4 * n} B | {us(added['none'][n])} | {us(added['centered'][n])} | "
              f"{us(added['centered_ranks'][n])} | {us(ladder[n]) if n in ladder else ''} |")

    lo, hi = pops[0], pops[-1]
    print(f"\nGrowth from N={lo} to N={hi}, the payload-dependent part of a gather:")
    for s in ("none", "centered", "centered_ranks"):
        print(f"  {s:15s} {us(added[s][hi] - added[s][lo]):>7} us")
    if ladder:
        small = min(ladder)
        print(f"  {'ladder':15s} {us(ladder[max(ladder)] - ladder[small]):>7} us"
              f"   (N={small} to N={max(ladder)})")
    print("\n`none` does not grow with a 4096-fold payload, so it is not paying for one. "
          "Read the gather from\nthe ladder, or from the two programs that have to "
          "consume what they gathered.")

    print(f"\n## The sort, read at D=1 where nothing is communicated, and the whole "
          f"barrier at D={d}\n")
    print(f"| N | sort us (ranks - none, D=1) | ns per member | whole ranks iteration, "
          f"D={d}, us |")
    print("|---|---|---|---|")
    for n in pops:
        sort = cells[(n, 1)]["centered_ranks"] - cells[(n, 1)]["none"]
        print(f"| {n} | {us(sort)} | {sort / n * 1e9:.1f} | "
              f"{us(cells[(n, d)]['centered_ranks'])} |")
    big = pops[-1]
    a, b = cells[(big, 1)]["centered_ranks"], cells[(big, d)]["centered_ranks"]
    print(f"\nAt N={big} the ranks program takes {us(a)} us on one device and {us(b)} on "
          f"{d}: {100 * (b - a) / a:.2f}% apart,\nand the {us(b - a)} us between them is "
          "the gather and the repeat noise, not a second sort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
