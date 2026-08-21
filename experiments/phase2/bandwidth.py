"""What the collective payload actually demands of the interconnect.

    python comms.py --config sweep-postfix.yaml --json comms.json
    python bandwidth.py --comms comms.json --results results-postfix

`comms.py` counts bytes per generation and `run.py` times the generation. Neither alone says
whether communication matters; the ratio does, and the ratio is what `docs/03` claims when it
says communication is a fraction of a percent of NVLink.

**This exists because that claim was, for one revision, an extrapolation.** The pre-fix
figure (1.894 GB/s) divided pre-fix bytes by pre-fix generation times. The sharding fix
changed both: strategy A's payload doubled, and generations got faster. Scaling the old
number by hand gives an estimate, and an estimate is not what `CLAUDE.md` ground rule 2 asks
a documented number to be. Given a post-fix sweep, this computes it.

Payload, not wire bytes. A ring all-reduce moves roughly `2(D-1)/D` times the payload on the
physical links, so the wire figure is a constant factor above this and depends on the
algorithm XLA picks. The payload is the part that is a property of the design, and it is a
lower bound on what the interconnect has to carry.

`D=1` is skipped: there is no interconnect to demand anything of.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: An A100's NVLink, the interconnect the sweep ran on. Quoted as the aggregate per-GPU
#: bidirectional figure, which is how the number is usually cited and how docs/03 cites it.
NVLINK_GBPS = 600.0

#: comms.py keys a configuration by the program, not by the sweep mode. Strong and weak
#: legitimately collide here: weak `N/device=32` at `D=8` is the same program as strong
#: `N=256` at `D=8`, so the same bytes and the same time. The collision is the two modes
#: agreeing, not an ambiguity.
KEY = ("d_model", "population", "strategy", "how", "devices")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--comms", type=pathlib.Path, required=True,
                    help="the --json output of comms.py")
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results-postfix")
    ap.add_argument("--top", type=int, default=10, help="how many rows to show")
    ap.add_argument("--reference", default=None,
                    help='link bandwidth as "NAME:GBPS", or "none" for GB/s only')
    args = ap.parse_args(argv)

    comms = json.loads(args.comms.read_text())
    times: dict[tuple, list[float]] = {}
    for p in sorted(args.results.glob("*.json")):
        r = json.loads(p.read_text())
        if "seconds_median" not in r:
            continue  # recorded error/OOM rows carry no time (TPU sweep has one)
        c = r["config"]
        times.setdefault(tuple(c[k] for k in KEY), []).append(r["seconds_median"])
    if not times:
        sys.exit(f"no results in {args.results}")

    rows, unmatched = [], 0
    for m in comms:
        if m["devices"] == 1:
            continue
        key = tuple(m[k] for k in KEY)
        seen = times.get(key)
        if not seen:
            unmatched += 1
            continue
        # Two modes producing the same program should produce the same time. Where they
        # differ, the slower one is the conservative choice: it makes the bandwidth demand
        # look smaller, and the claim being tested is that the demand is small.
        secs = max(seen)
        rows.append((m["bytes"] / secs / 1e9, m, secs, len(seen)))

    if not rows:
        sys.exit("no configuration appears in both the comms table and the results")
    rows.sort(reverse=True)

    kind = json.loads(next(args.results.glob("*.json")).read_text())["env"]["device_kind"]
    # The percentage column needs the platform's link bandwidth; NVLink's 600 is
    # only right for the A100 runs. --reference "NAME:GBPS" overrides it, and
    # --reference none prints the demanded GB/s alone, for a platform whose link
    # figure should be cited rather than hardcoded (the v5e's ICI).
    ref_name, ref_gbps = "an A100's ~600 GB/s NVLink", NVLINK_GBPS
    if args.reference == "none":
        ref_gbps = None
    elif args.reference:
        ref_name, g = args.reference.rsplit(":", 1)
        ref_gbps = float(g)

    print(f"{args.results.name}: {kind}")
    print(f"{len(rows)} multi-device configurations matched; {unmatched} in the comms table "
          f"had no timing\n")
    print(f"{'d':>6}{'N':>6}  {'strategy':17}{'how':>4}{'D':>3}"
          f"{'bytes/gen':>12}{'ms/gen':>9}{'GB/s':>9}"
          + (f"{'% link':>10}" if ref_gbps else ""))
    for gbps, m, secs, _ in rows[:args.top]:
        pct = f"{100 * gbps / ref_gbps:>9.2f}%" if ref_gbps else ""
        print(f"{m['d_model']:>6}{m['population']:>6}  {m['strategy']:17}{m['how']:>4}"
              f"{m['devices']:>3}{m['bytes']:>12,}{secs * 1e3:>9.2f}{gbps:>9.3f}{pct}")

    worst, m, secs, _ = rows[0]
    print(f"\nmost demanding: d={m['d_model']} N={m['population']} {m['strategy']}/{m['how']} "
          f"at D={m['devices']}")
    if ref_gbps:
        print(f"  {worst:.3f} GB/s, {100 * worst / ref_gbps:.2f}% of {ref_name}")
    else:
        print(f"  {worst:.3f} GB/s demanded; divide by the platform's published "
              "link bandwidth, cited, for the percentage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
