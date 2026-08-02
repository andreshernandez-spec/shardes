#!/usr/bin/env python
"""The guard that has to pass before any scaling number is quoted.

    python check.py --results results-rehearsal      # non-zero exit if anything diverged

`docs/03`: "Fixed seed; assert the optimizer trajectory is identical across D" — because a
scaling number for two different computations is not a scaling number. This is that
assertion, run over whatever a sweep wrote.

**Identical does not mean bitwise, and which one it is depends on the contraction strategy.**
Strategy A regenerates and contracts the whole population on every device in the same order,
so the update is bitwise identical at D=1 and D=8. Strategy B `psum`s a partial update per
device, so the summation order *is* the device count. B is expected to differ in the last
ulp and A is not, so the two are held to different standards here: A must match exactly, B
must match within `RTOL`.

Holding B to exact equality would fail forever; holding A to a tolerance would hide a real
bug. Both were live possibilities before the rehearsal measured it.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

#: One generation, same platform, different device counts. Matches
#: tests/gpu/test_device_invariance_gpu.py's SHARDING_RTOL, deliberately: same claim.
RTOL = 1e-5


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error in the L2 norm, the same measure `tests/gpu` uses.

    **Not elementwise `max(|a-b|/|b|)`, which is what this used to be and which was wrong.**
    Dividing componentwise means any component of `b` near zero inflates the ratio without
    bound, and a random projection of a parameter vector has components near zero all the
    time. That metric reported `lowrank_r1/B` diverging by 2.7e-02 on a run where the real
    disagreement was 2.3e-07, and it failed a TPU prep run whose cause was then misattributed.

    The norm ratio asks the question actually being asked: is the *update* the same vector,
    whatever happened to individual coordinates.
    """
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def load(results: pathlib.Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    if not rows:
        sys.exit(f"no results in {results}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    args = ap.parse_args(argv)

    rows = load(args.results)
    errors = [r for r in rows if "error" in r]

    groups: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
    for r in rows:
        c = r["config"]
        if "error" in r or c["mode"] != "strong":
            continue  # weak scaling changes N with D on purpose, so it has no such claim
        key = (c["d_model"], c["population"], c["strategy"], c["how"])
        groups[key][c["devices"]] = r["trajectory"]

    print(f"{'d':>5} {'N':>6} {'strategy':18s} {'how':4s} {'D':>12}  exact  max rel dev")
    failures = []
    for key, by_devices in sorted(groups.items()):
        d_model, population, strategy, how = key
        base = by_devices[min(by_devices)]
        exact = len({t["digest"] for t in by_devices.values()}) == 1
        worst = 0.0
        for t in by_devices.values():
            a, b = np.array(t["probe"]), np.array(base["probe"])
            worst = max(worst, _rel(a, b))

        if how == "A" and not exact:
            failures.append(f"{strategy}/A is not bitwise identical across D ({worst:.2e})")
        if how == "B" and worst > RTOL:
            failures.append(f"{strategy}/B exceeds rtol {RTOL:g} across D ({worst:.2e})")

        devices = ",".join(str(d) for d in sorted(by_devices))
        print(
            f"{d_model:>5} {population:>6} {strategy:18s} {how:4s} {devices:>12}  "
            f"{str(exact):5s}  {worst:.2e}"
        )

    print()
    if errors:
        print(f"{len(errors)} configuration(s) recorded an error:")
        for r in errors[:5]:
            print(f"  {r['config']} -> {r['error']}")
    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        print("\nA scaling number from these results would compare different computations.")
        return 1
    print(f"OK: {len(groups)} strong-scaling groups, every device count ran the same thing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
