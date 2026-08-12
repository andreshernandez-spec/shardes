"""Two sweeps of the same configurations, before and after a change.

    python compare.py --before results --after results-postfix

Written for the M1 re-run: `results/` is the 2026-08-06 sweep at commit 953283f, before
`ShardedES.apply` constrained its output to the member axis, and `results-postfix/` is the
same configurations after. It is not specific to that change; anything that is supposed to
alter speed without altering arithmetic can be checked this way.

**What makes the comparison legitimate is that the commit is the ONLY thing allowed to
differ.** `check.py` puts `commit` in `COMPARABLE`, because within one sweep two device
counts built from different libraries have not run the same computation. Here the commit is
the independent variable, so it is excluded and *everything else* is held: platform, chip,
jax, jaxlib and the XLA flags. A pair that differs in anything else is refused rather than
reported, because the difference could not then be attributed to the change.

**The digest check is exact at `D=1` and cannot be at `D>1`, and that asymmetry is not a
loophole.** This tool first demanded bitwise equality everywhere. That is wrong, and the
sharding fix is itself the reason.

Before the fix the evaluation was replicated, so every device count ran the *same program
shape*: a vmap over all `N` members, `D` times over. `check.py` duly reports strategy A as
`0.00e+00` across `D=1,2,4,8` for the whole pre-fix sweep, and that pass was vacuous. After
the fix, `D=4` vmaps over `N/4`. A different shape reduces in a different order, so float32
lands a fraction of an ulp elsewhere. Requiring the sharded program to be bit-identical to
the replicated one is requiring the fix not to have happened.

So:

  - `D=1` must be bitwise identical. The constraint is a no-op there, the program shape is
    unchanged, and a moved digest means something other than the change under test moved.
    This is a hard failure.
  - `D>1` is reported, never asserted. The update's relative error is printed next to
    whether the *ranks* moved, which is the distinction `rank_digest` exists to draw: ranks
    holding means summation order alone, ranks moving means the population is packed tightly
    enough that an ulp reorders it and the rank shaping amplifies that into the update. The
    second is the documented noise floor, a property of the configuration.

Judging `D>1` belongs to `check.py` run against the new results, which asks the question
that is actually answerable: is the post-fix program self-consistent across device counts.

A baseline predating `rank_digest` records no ranks, and then neither case can be shown.
That is reported as unadjudicated rather than quietly counted as a pass.

Exit status is 1 if any pair fails, so this can gate a docs update.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

#: Everything `check.py` holds constant EXCEPT `commit`. See the module docstring: the commit
#: is what the two sweeps differ by, and holding it would refuse every pair.
COMPARABLE = ("device_platform", "device_kind", "jax", "jaxlib", "xla_flags")

#: A config is identified by what was asked for, not by filename, so a rename or an added
#: field cannot silently pair two different configurations.
KEY = ("mode", "d_model", "population", "devices", "strategy", "how")


def load(results: pathlib.Path) -> dict[tuple, dict]:
    rows = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    if not rows:
        sys.exit(f"no results in {results}")
    out = {}
    for r in rows:
        out[tuple(r["config"][k] for k in KEY)] = r
    return out


def relative(a: dict, b: dict) -> float:
    """Relative L2 error between two trajectory probes.

    The norm ratio, not elementwise, for the reason `check.py._rel` spells out: a random
    projection has components near zero all the time, and dividing by them reports a
    divergence that is not there.
    """
    pa = np.asarray(a["trajectory"]["probe"], dtype=np.float64)
    pb = np.asarray(b["trajectory"]["probe"], dtype=np.float64)
    denom = float(np.linalg.norm(pb))
    return float(np.linalg.norm(pa - pb) / denom) if denom else float(np.linalg.norm(pa - pb))


def environment(row: dict) -> tuple:
    env = row.get("env", {})
    return tuple(str(env.get(k, "?")) for k in COMPARABLE)


def efficiency(rows: dict[tuple, dict]) -> dict[tuple, float]:
    """T1/(D*T_D) per series, keyed by the config of the D-device member.

    The baseline is the same series at D=1. A series with no D=1 result has no efficiency,
    not an efficiency of 1: strong scaling is defined against one device and nothing else.
    """
    base = {}
    for key, row in rows.items():
        mode, d, n, devices, strategy, how = key
        if devices == 1:
            base[(mode, d, n, strategy, how)] = row["seconds_median"]
    out = {}
    for key, row in rows.items():
        mode, d, n, devices, strategy, how = key
        t1 = base.get((mode, d, n, strategy, how))
        if t1 is None:
            continue
        out[key] = t1 / (devices * row["seconds_median"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--before", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--after", type=pathlib.Path, default=HERE / "results-postfix")
    ap.add_argument("--quiet", action="store_true", help="verdict only, no per-config table")
    args = ap.parse_args(argv)

    before, after = load(args.before), load(args.after)
    shared = sorted(set(before) & set(after))
    if not shared:
        sys.exit("no configurations in common; these two sweeps do not compare")

    print(f"{args.before.name}: {len(before)} configs")
    print(f"{args.after.name}: {len(after)} configs")
    print(f"shared: {len(shared)}")
    only_before, only_after = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    # Reported rather than ignored. A re-run that silently covered half the configurations
    # would otherwise pass every check it did run.
    if only_before:
        print(f"only in {args.before.name}: {len(only_before)} (not compared)")
    if only_after:
        print(f"only in {args.after.name}: {len(only_after)} (not compared)")

    commits = {r["env"].get("commit", "?")[:7] for r in before.values()} | set()
    print(f"\nbefore commit(s): {', '.join(sorted(commits))}")
    print(f"after commit(s):  {', '.join(sorted({r['env'].get('commit', '?')[:7] for r in after.values()}))}")

    incomparable, digest_bad, dirty, moved = [], [], [], []
    for key in shared:
        if environment(before[key]) != environment(after[key]):
            diff = "; ".join(
                f"{name}: {environment(before[key])[i]} vs {environment(after[key])[i]}"
                for i, name in enumerate(COMPARABLE)
                if environment(before[key])[i] != environment(after[key])[i]
            )
            incomparable.append((key, diff))
        if before[key]["trajectory"]["digest"] != after[key]["trajectory"]["digest"]:
            # Only D=1 is a failure. See the module docstring: at D>1 the post-fix program
            # is a different shape by design and cannot be bit-identical to the replicated
            # one it replaces.
            (digest_bad if key[3] == 1 else moved).append(key)
        for label, row in (("before", before[key]), ("after", after[key])):
            if row["env"].get("dirty_worktree"):
                dirty.append((key, label))

    eff_before, eff_after = efficiency(before), efficiency(after)

    if not args.quiet:
        print(f"\n{'config':<52} {'eff before':>11} {'eff after':>10} {'speedup':>8}  digest")
        for key in shared:
            mode, d, n, devices, strategy, how = key
            if devices == 1:
                continue
            eb, ea = eff_before.get(key), eff_after.get(key)
            name = f"{mode} d={d} N={n} {strategy}/{how} D={devices}"
            speed = before[key]["seconds_median"] / after[key]["seconds_median"]
            ok = "ok" if before[key]["trajectory"]["digest"] == after[key]["trajectory"]["digest"] else "MOVED"
            eb_s = f"{eb:.3f}" if eb is not None else "-"
            ea_s = f"{ea:.3f}" if ea is not None else "-"
            print(f"{name:<52} {eb_s:>11} {ea_s:>10} {speed:>7.2f}x  {ok}")

    print()
    if dirty:
        print(f"WARNING: {len(dirty)} result(s) recorded a dirty worktree; provenance is not exact")
    if incomparable:
        print(f"REFUSED: {len(incomparable)} pair(s) differ in more than the commit")
        for key, diff in incomparable[:5]:
            print(f"  {key}: {diff}")
        print("  the difference cannot be attributed to the change under test")
    if digest_bad:
        print(f"FAIL: {len(digest_bad)} D=1 trajectory digest(s) moved")
        for key in digest_bad[:8]:
            print(f"  {key}")
        print("  D=1 runs the same program before and after; the constraint is a no-op")
        print("  there. Something other than the change under test differs, and this is")
        print("  not a before/after of one computation.")
    if moved:
        print(f"{len(moved)} multi-device update(s) moved. Expected: a sharded evaluation")
        print("reduces in a different order than the replicated one it replaces.")
        print(f"\n  {'config':<44}{'relerr':>10}  ranks")
        unadjudicated = 0
        for key in moved:
            ra = before[key].get("ranks", {}).get("digest")
            rb = after[key].get("ranks", {}).get("digest")
            if ra is None or rb is None:
                verdict, unadjudicated = "not recorded before this baseline", unadjudicated + 1
            elif ra == rb:
                verdict = "held: summation order only"
            else:
                verdict = f"MOVED: noise floor, {after[key].get('ranks', {}).get('ties', '?')} ties"
            name = f"{key[0]} d={key[1]} N={key[2]} {key[4]}/{key[5]} D={key[3]}"
            print(f"  {name:<44}{relative(before[key], after[key]):>10.2e}  {verdict}")
        if unadjudicated:
            print(f"\n  {unadjudicated} could not be adjudicated: the baseline predates")
            print("  rank_digest, so it does not record whether ranks moved. Do not read")
            print("  these as passes.")
        print("\n  check.py --results <after> is what judges the new run's self-consistency.")
    if not (incomparable or digest_bad):
        # Printed unconditionally. This used to live inside the `if multi` below, so a
        # partial run of D=1 results passed the digest check and then said nothing at all,
        # which reads exactly like a checker that did not run.
        # Counts what was actually asserted, not what was compared. Saying "N digests
        # identical" while N-7 were identical is the kind of summary line that gets quoted
        # into a doc and then defended.
        one = [k for k in shared if k[3] == 1]
        exact = len(one) - len(digest_bad)
        print(f"OK: environment held except commit; {exact}/{len(one)} D=1 digests exact"
              + (f", {len(moved)} multi-device updates moved (above)" if moved else ""))
        # Averaged over the D>1 members only. D=1 has efficiency 1 by construction and
        # including it would dilute the number toward 1 for free.
        multi = [k for k in shared if k[3] > 1 and k in eff_before and k in eff_after]
        if multi:
            gb = sum(eff_before[k] for k in multi) / len(multi)
            ga = sum(eff_after[k] for k in multi) / len(multi)
            print(f"    mean parallel efficiency over {len(multi)} multi-device configs: "
                  f"{gb:.3f} -> {ga:.3f}")
        else:
            print("    no multi-device configuration in common yet, so no efficiency to report")
    return 1 if (incomparable or digest_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
