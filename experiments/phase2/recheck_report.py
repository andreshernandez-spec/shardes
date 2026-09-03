#!/usr/bin/env python
"""The three questions the 2026-09-03 8x A100 session was rented to answer.

    python recheck_report.py

1. How much does a cell move between *runs*, not between repeats inside one?
   The sweep times consecutive generations in one process after one compile, so
   its own spread cannot see compilation or process variation. Five separate
   invocations of the same committed config can.
2. Do the two residuals that survived the noise on the original host survive on
   a different one?
3. Does a resident population-sized buffer slow the replicated contraction?
   That was the leading explanation for those residuals: `iid_gaussian` is the
   only arm that materializes its population, so A's contraction may run while
   that memory is still occupied, which the isolation harness cannot see.

Reads `results-recheck-2026-09-03/`. The model is `timemodel.py`'s, evaluated
against this session's own isolation records rather than the committed ones.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import timemodel  # noqa: E402

SESSION = HERE / "results-recheck-2026-09-03"
RUNS = [f"run{i}" for i in range(1, 6)]
KIND = "nvidia-a100-sxm4-80gb"
CELLS = ((512, 256), (512, 1024))
ALL_CELLS = ((512, 256), (512, 1024), (2048, 128), (2048, 256))


def sweep_median(n: int, how: str, run: str) -> float:
    f = SESSION / run / f"mode=strong__D=8__d=512__N={n}__s=iid_gaussian__how={how}.json"
    return json.loads(f.read_text())["seconds_median"]


def isolation(d: int, n: int, which: str) -> dict:
    tag = "__resident" if which == "contraction-resident" else ""
    f = SESSION / which / f"d={d}__N={n}__s=iid_gaussian__{KIND}__D8{tag}.json"
    return json.loads(f.read_text())


def main() -> int:
    fab = timemodel.Fabric.from_ladder(
        HERE / "results-ladder" / f"ladder-{KIND}-D8.json", "A100")

    print("## 1. Between-run spread, five fresh invocations of the committed config\n")
    print("| cell | placement | run medians ms | spread ms | sd ms |")
    print("|---|---|---|---|---|")
    for _, n in CELLS:
        for how in "AB":
            m = [sweep_median(n, how, r) * 1e3 for r in RUNS]
            print(f"| d=512 N={n} | {how} | {', '.join(f'{x:.3f}' for x in m)} | "
                  f"{max(m) - min(m):.3f} | {statistics.stdev(m):.3f} |")

    print("\n## 2. Do the residuals reproduce on a different host?\n")
    print("| cell | measured A-B ms | predicted ms | residual ms | shard ratio |")
    print("|---|---|---|---|---|")
    for d, n in CELLS:
        delta = (statistics.median([sweep_median(n, "A", r) for r in RUNS])
                 - statistics.median([sweep_median(n, "B", r) for r in RUNS]))
        rec = isolation(d, n, "contraction-plain")
        pred = (rec["contraction_seconds"] - rec["contraction_local_seconds"]
                + fab.allgather(4 * n) - rec["allreduce_insitu_seconds"])
        print(f"| d={d} N={n} | {delta * 1e3:+.3f} | {pred * 1e3:+.3f} | "
              f"{(delta - pred) * 1e3:+.3f} | {rec['shard_ratio']:.3f} |")

    print("\n## 3. Does a resident population-sized buffer slow the contraction?\n")
    print("| cell | C plain ms | C resident ms | change ms | change % | resident MiB/dev |")
    print("|---|---|---|---|---|---|")
    for d, n in ALL_CELLS:
        p = isolation(d, n, "contraction-plain")
        r = isolation(d, n, "contraction-resident")
        dc = r["contraction_seconds"] - p["contraction_seconds"]
        print(f"| d={d} N={n} | {p['contraction_seconds'] * 1e3:.3f} | "
              f"{r['contraction_seconds'] * 1e3:.3f} | {dc * 1e3:+.3f} | "
              f"{100 * dc / p['contraction_seconds']:+.2f} | "
              f"{r.get('resident_bytes_per_device', 0) / 2**20:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
