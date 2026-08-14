#!/usr/bin/env python
"""The guard that has to pass before any scaling number is quoted.

    python check.py --results results-rehearsal      # non-zero exit if anything diverged

`docs/03`: "Fixed seed; assert the optimizer trajectory is identical across D", because a
scaling number for two different computations is not a scaling number. This is that
assertion, run over whatever a sweep wrote.

**Identical does not mean bitwise, and which one it is depends on the contraction strategy
and on the shaping.** Strategy A regenerates and contracts the whole population on every
device in the same order; strategy B `psum`s a partial update per device, so the summation
order *is* the device count. A must match exactly, B within `RTOL`.

Holding B to exact equality would fail forever; holding A to a tolerance would hide a real
bug. Both were live possibilities before the rehearsal measured it.

**A's exactness is not free-standing, and an 8x A100 run is what showed why.** The fitness is
*not* bitwise identical across device counts even under A: 1024 members in one batch and 128
on each of eight are different shapes, and XLA reduces different shapes with different trees.
Measured at `d=512, N=1024`: 189 of 1024 entries differ, by up to 2.00 ulp.
`--xla_gpu_deterministic_ops=true` does not help, because it fixes the algorithm for a given
shape rather than across shapes.

What makes A exact is `centered_ranks`. The update is a function of the ordering alone, so a
sub-rank difference never reaches it, and every group that passes carries that same ulp of
disagreement. Two things follow, and both are enforced below rather than assumed:

- the exact standard is conditional on a rank shaping, so `shaping` is recorded and
  `RANK_SHAPINGS` gates it. Under `centered` or `none` the fitness reaches the update and A
  is held to B's tolerance instead;
- a configuration whose members are close enough for an ulp to reorder them fails A with
  nothing wrong in the contraction. That is the noise floor, `ranks` is how this tells it
  from a real bug, and `noisefloor.py` predicts it.

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

#: Shapings whose output depends only on the *ordering* of the fitnesses. Under one of these
#: an ulp of arithmetic difference is discarded unless it moves a rank, which is the entire
#: reason strategy A comes out bitwise identical across device counts. Under `centered` or
#: `none` the fitness reaches the update directly and A is expected to differ in the last
#: ulp exactly like B, so holding it to exact equality would fail every group.
#:
#: Measured on 8x A100: the fitness is *not* bitwise identical across device counts even
#: under A (189 of 1024 entries at `d=512, N=1024`). Every passing group has that same
#: sub-rank difference and the rank transform throws it away.
RANK_SHAPINGS = ("centered_ranks",)


def _environment(row: dict) -> tuple:
    env = row.get("env", {})
    return tuple(str(env.get(k, "?")) for k in COMPARABLE) + (
        str(row.get("guard_precision", "?")),
        # Two device counts shaped differently did not run the same computation, so this
        # belongs with the environment rather than as a separate check.
        str(row.get("shaping", "?")),
    )


def _fields() -> tuple:
    return COMPARABLE + ("guard_precision", "shaping")


def _exactness_expected(rows: list[dict]) -> bool:
    """Is strategy A supposed to be bitwise identical for these rows?

    Only under a rank shaping. Results written before `shaping` was recorded do not say, and
    every sweep that produced them ran the `centered_ranks` default, so absent means yes:
    the alternative is invalidating results that are still perfectly good.
    """
    seen = {r.get("shaping") for r in rows}
    if seen == {None}:
        return True
    return all(s in RANK_SHAPINGS for s in seen if s is not None)


def _ranks_moved(by_devices: dict) -> bool | None:
    """Did any member change rank across device counts? None when nothing recorded it."""
    digests = {d: r.get("ranks", {}).get("digest") for d, r in by_devices.items()}
    if any(v is None for v in digests.values()):
        return None
    return len(set(digests.values())) > 1


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
    ap.add_argument("--config", type=pathlib.Path, default=None,
                    help="the sweep config these results should cover. Given it, every "
                         "expected configuration must be present; without it, this can only "
                         "check what happens to be here")
    args = ap.parse_args(argv)

    rows = load(args.results)
    errors = [r for r in rows if "error" in r]

    # **What is absent is invisible without the config.** Every check below reads the rows
    # that exist, so a sweep that wrote half its device counts, or died before `d=2048`, or
    # was pointed at the wrong directory, passes on the half it managed. Only the config
    # knows what should be there.
    missing = []
    if args.config:
        import sys as _sys
        _sys.path.insert(0, str(HERE))
        import run as R  # the one enumeration, so the checker cannot drift from the driver
        cfg = R.yaml.safe_load(args.config.read_text())
        have = {(r["config"]["mode"], r["config"]["d_model"], r["config"]["population"],
                 r["config"]["devices"], r["config"]["strategy"], r["config"]["how"])
                for r in rows}
        for c in R.expand(cfg):
            key = (c.mode, c.d_model, c.population, c.devices, c.strategy, c.how)
            if key not in have:
                missing.append(key)

    groups: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
    for r in rows:
        c = r["config"]
        if "error" in r or c["mode"] != "strong":
            continue  # weak scaling changes N with D on purpose, so it has no such claim
        key = (c["d_model"], c["population"], c["strategy"], c["how"])
        groups[key][c["devices"]] = r

    print(f"{'d':>5} {'N':>6} {'strategy':18s} {'how':4s} {'D':>12}  exact  max rel dev  ranks")
    failures = []
    floored = []
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

        moved = _ranks_moved(by_devices)

        trajectories = {d: r["trajectory"] for d, r in by_devices.items()}
        base = trajectories[min(trajectories)]
        exact = len({t["digest"] for t in trajectories.values()}) == 1
        worst = 0.0
        for t in trajectories.values():
            a, b = np.array(t["probe"]), np.array(base["probe"])
            worst = max(worst, _rel(a, b))

        where = f"{strategy}/{how} d={d_model} N={population}"
        if how == "A" and not exact:
            if not _exactness_expected(here):
                # Not a rank shaping, so the fitness reaches the update and A is no more
                # exact than B. Hold it to B's standard rather than to one it cannot meet.
                shaping = sorted({str(r.get("shaping")) for r in here})
                if worst > RTOL:
                    failures.append(
                        f"{where} exceeds rtol {RTOL:g} across D ({worst:.2e}) under "
                        f"shaping {'/'.join(shaping)}"
                    )
            elif moved is False:
                # The discriminator. Same ordering, different update, under a shaping that
                # reads nothing but the ordering. Nothing about the noise floor explains it.
                failures.append(
                    f"{where} is not bitwise identical across D ({worst:.2e}) with the "
                    "SAME ranks on every device count, so the contraction itself differs"
                )
            elif moved is True:
                floored.append((where, worst))
            else:
                failures.append(
                    f"{where} is not bitwise identical across D ({worst:.2e}); no rank "
                    "digest was recorded, so re-run to tell a contraction bug from the "
                    "noise floor"
                )
        if how == "B" and worst > RTOL:
            # B is allowed its last ulp, so exceeding RTOL is either a real problem or the
            # same reordering that breaks A. Same discriminator, same split.
            if moved is True:
                floored.append((where, worst))
            else:
                failures.append(f"{where} exceeds rtol {RTOL:g} across D ({worst:.2e})")

        print(
            f"{d_model:>5} {population:>6} {strategy:18s} {how:4s} {devices:>12}  "
            f"{str(exact):5s}  {worst:.2e}  "
            f"{'?' if moved is None else ('MOVED' if moved else 'same')}"
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

    for f in failures:
        print(f"FAIL: {f}")

    if floored:
        # Separated from FAIL on purpose. These configurations did compute different things
        # across device counts, so no scaling number may be quoted from them, but the cause
        # is the population rather than the library and the response is to write it down.
        print(f"\nNOISE FLOOR: {len(floored)} group(s) reordered their population across "
              "device counts.")
        for where, worst in floored:
            print(f"  {where} ({worst:.2e})")
        print()
        print("Members changed rank, so the update changed. The fitness differs by an ulp or")
        print("so at every device count, including the groups that pass; a rank transform")
        print("normally discards that, and here the population is packed tightly enough that")
        print("it does not. This is a property of the configuration, not a contraction bug.")
        print()
        print("  python noisefloor.py --config <the sweep config>")
        print()
        print("Quote no scaling number from these groups without saying which they are.")

    if errors:
        print(f"\n{len(errors)} configuration(s) recorded an error rather than a measurement:")
        for r in errors[:5]:
            c = r["config"]
            print(f"  d={c['d_model']} N={c['population']} {c['strategy']}/{c['how']} "
                  f"D={c['devices']}: {r['error'][:80]}")
        print("  Errors used to print here and still return 0. They do not any more.")
    if missing:
        print(f"\n{len(missing)} configuration(s) from {args.config.name} have no result:")
        for k in missing[:5]:
            print(f"  {k}")
        print("  A checker that only reads what is present cannot see a sweep that stopped.")

    # **Three severities, not one exit code.** This returned 1 for errors, missing rows and
    # the noise floor alike, which makes it useless as a mid-session gate: the noise floor is
    # a permanent property of six `d=512` shapes and would block every run forever, so a
    # session that gated on this would never proceed and a session that ignored it would
    # ignore the errors too.
    #
    #   2  the run is broken: a configuration errored, or the matrix has holes
    #   1  the run is complete but a scaling number from some group would be misleading
    #   0  nothing to report
    #
    # A caller gating a rented session should stop on 2 and carry on past 1, having read
    # which groups were named.
    if errors or missing:
        print("\nEXIT 2: this run is incomplete. Fix or re-run before quoting anything.")
        return 2
    if failures or floored:
        print("\nA scaling number from these results would compare different computations.")
        return 1
    covered = f", covering all of {args.config.name}" if args.config else ""
    print(f"OK: {len(groups)} strong-scaling groups, every device count ran the same "
          f"thing{covered}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
