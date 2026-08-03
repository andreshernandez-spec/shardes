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

**The comparison is only valid within one execution environment, and that is now enforced
rather than assumed.** It used to be a comment on `RTOL` saying "same platform", which is
a precondition nobody checked. The driver resumes by skipping configurations already on
disk, so a sweep started on one backend and finished on another silently produces a
results directory whose `D=1` and `D=2` rows never ran the same arithmetic. That happened:
`lowrank_r1` was read as violating device-count invariance at 6.33e-03 when the `D=1` row
had come from a GPU and the `D=2` row from CPU. Re-run on one backend it was bitwise
identical. A guard that can be fooled by its own resume feature is not a guard.
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

#: What has to match before two device counts are comparable at all. Each of these can
#: change the arithmetic: the backend and the chip pick different kernels, a JAX version
#: changes what those kernels are, and a different commit is a different library.
#:
#: `hostname` is deliberately absent. Two identical nodes are comparable, and requiring it
#: would fail a legitimate resume, which is the feature this is meant to protect rather
#: than punish.
#: `xla_flags` is in here for a measured reason. Without
#: `--xla_gpu_deterministic_ops=true` XLA picks reduction algorithms per shape, so D=1 and
#: D=2 run different arithmetic; on 2x T4 that produced a 6.3e-03 disagreement on a
#: configuration that is exactly zero with the flag set. Comparing a flagged run against an
#: unflagged one is as meaningless as comparing CPU against GPU.
COMPARABLE = ("device_platform", "device_kind", "jax", "jaxlib", "commit", "xla_flags")


def _environment(row: dict) -> tuple:
    env = row.get("env", {})
    return tuple(str(env.get(k, "?")) for k in COMPARABLE) + (
        str(row.get("guard_precision", "?")),
    )


def _fields() -> tuple:
    return COMPARABLE + ("guard_precision",)


def _mismatch(rows: list[dict]) -> str:
    """Only the fields that actually differ, so the failure line names the cause."""
    out = []
    for i, name in enumerate(_fields()):
        seen = sorted({_environment(r)[i] for r in rows})
        if len(seen) > 1:
            out.append(f"{name}: {' vs '.join(seen)}")
    return "; ".join(out)


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
        groups[key][c["devices"]] = r

    print(f"{'d':>5} {'N':>6} {'strategy':18s} {'how':4s} {'D':>12}  exact  max rel dev")
    failures = []
    for key, by_devices in sorted(groups.items()):
        d_model, population, strategy, how = key
        here = list(by_devices.values())
        devices = ",".join(str(d) for d in sorted(by_devices))

        # Comparability first. A number computed across two environments is not a
        # disagreement between device counts, and reporting it as one sends the
        # investigation at the library instead of at the results directory.
        if len({_environment(r) for r in here}) > 1:
            failures.append(
                f"{strategy}/{how} d={d_model} N={population} spans environments, "
                f"so its device counts are not comparable ({_mismatch(here)})"
            )
            print(
                f"{d_model:>5} {population:>6} {strategy:18s} {how:4s} {devices:>12}  "
                f"{'-':5s}  MIXED ENV"
            )
            continue

        trajectories = {d: r["trajectory"] for d, r in by_devices.items()}
        base = trajectories[min(trajectories)]
        exact = len({t["digest"] for t in trajectories.values()}) == 1
        worst = 0.0
        for t in trajectories.values():
            a, b = np.array(t["probe"]), np.array(base["probe"])
            worst = max(worst, _rel(a, b))

        if how == "A" and not exact:
            failures.append(f"{strategy}/A is not bitwise identical across D ({worst:.2e})")
        if how == "B" and worst > RTOL:
            failures.append(f"{strategy}/B exceeds rtol {RTOL:g} across D ({worst:.2e})")

        print(
            f"{d_model:>5} {population:>6} {strategy:18s} {how:4s} {devices:>12}  "
            f"{str(exact):5s}  {worst:.2e}"
        )

    print()
    # Every environment the directory contains, so a mixed sweep is visible even when no
    # single group straddles the boundary.
    seen = sorted({(r.get("env", {}).get("device_kind", "?"),
                    r.get("env", {}).get("commit", "?")[:9]) for r in rows if "error" not in r})
    if len(seen) > 1:
        print(f"{len(seen)} environments in this directory:")
        for kind, commit in seen:
            print(f"  {kind} @ {commit}")
        print()

    # Not a failure: `docs/03` wants results traceable to a commit, and run.py already warns
    # at write time. Repeating it here is what makes it visible to whoever reads the guard.
    dirty = [r for r in rows if r.get("env", {}).get("dirty_worktree")]
    if dirty:
        print(f"note: {len(dirty)}/{len(rows)} results were written from a dirty worktree "
              "and are not reproducible from a commit alone\n")

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
