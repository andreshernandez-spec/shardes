#!/usr/bin/env python
"""Answer Gate G0 from results/. The comparison, not the prose.

    python gate.py                 # the table, at the default sigma slice
    python gate.py --sigma 0.001   # the slice where the estimator has most signal
    python gate.py --all-sigma

G0 (docs/01): *do rank-1 estimator-quality curves separate across sampling schemes at
N/d_eff >~ 1, when full-rank curves at N/d_eff << 1 do not?*

Answering it needs a **matched pair**: same rank, same scheme family, same shaping, differing
only in the coupling. `mirrored_lr1` vs `mirrored_hd_lr1` is such a pair. Comparing
`lowrank_r1` against `mirrored_hd_lr1` is not, and would conflate coupling with mirroring and
with the shaping arm that mirroring implies.

"Separated" here means the two IQRs over R replicates do not overlap. That is deliberately
weak: with R = 30 it will call a real 2% effect ambiguous. A gate that a null has to clear
should be easy to clear, so that failing it means something.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"

# (uncoupled, coupled) at matched rank, scheme family and shaping arm. The shaping is the
# baseline arm for that scheme (plot.py BASELINE): `none` under mirroring, `centered` on the
# iid side.
PAIRS = [
    ("full rank", "mirrored_full", "mirrored_hd_full", "none"),
    ("rank 1", "mirrored_lr1", "mirrored_hd_lr1", "none"),
    ("rank 1", "mirrored_lr1", "mirrored_sobol_lr1", "none"),
    ("rank 4", "mirrored_lr4", "mirrored_hd_lr4", "none"),
    ("rank 4", "mirrored_lr4", "mirrored_sobol_lr4", "none"),
]


def load() -> dict:
    idx = {}
    for p in RESULTS.glob("*.json"):
        r = json.loads(p.read_text())
        c = r["config"]
        idx[(c["strategy"], c["population"], c["sigma"], c["shaping"])] = r
    return idx


def separated(a: dict, b: dict) -> bool:
    """IQRs disjoint. Weak on purpose; see the module docstring."""
    return a["cosine_q3"] < b["cosine_q1"] or b["cosine_q3"] < a["cosine_q1"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=0.001)
    ap.add_argument("--all-sigma", action="store_true")
    args = ap.parse_args(argv)

    idx = load()
    if not idx:
        print(f"no results in {RESULTS}", file=sys.stderr)
        return 1
    sigmas = sorted({k[2] for k in idx}) if args.all_sigma else [args.sigma]

    any_sep = False
    missing = []
    for sigma in sigmas:
        # sigma = 0.1 is a dead arm on this block: cos ~ 1e-3, occasionally negative. Ratios
        # there are noise over noise and are reported but never counted as separation.
        dead = sigma >= 0.1
        print(f"\n{'=' * 78}\nsigma = {sigma:g}{'   (DEAD ARM: cos ~ 1e-3, ratios are noise)' if dead else ''}")
        for panel, plain, coupled, shaping in PAIRS:
            ns = sorted({k[1] for k in idx if k[0] == plain and k[2] == sigma})
            rows = []
            for n in ns:
                a = idx.get((plain, n, sigma, shaping))
                b = idx.get((coupled, n, sigma, shaping))
                if not (a and b):
                    missing.append((coupled, n, sigma))
                    continue
                sep = separated(a, b) and not dead
                any_sep |= sep
                rows.append((n, n / a["d_eff"], a["cosine_median"], b["cosine_median"], sep))
            if not rows:
                continue
            print(f"\n  {panel}: {coupled} vs {plain}  (shaping={shaping})")
            print(f"  {'N':>8}{'N/d_eff':>10}{'plain':>11}{'coupled':>11}{'ratio':>9}  verdict")
            for n, ratio, ca, cb, sep in rows:
                print(f"  {n:>8}{ratio:>10.3f}{ca:>11.5f}{cb:>11.5f}{cb / ca:>9.4f}"
                      f"  {'SEPARATED' if sep else 'overlapping'}")

    print(f"\n{'=' * 78}")
    if missing:
        print(f"{len(missing)} cell(s) missing; the sweep is incomplete.")
    print(f"G0: any scheme separated at any N, rank or sigma?  {'YES' if any_sep else 'NO'}")
    if not any_sep:
        print("\nA null here is a real result, not a failed experiment (docs/01, Gate G0).")
        print("Before reading it as one, check the treatment was applied: see")
        print("tests/test_coupling.py::test_hd_block_is_exactly_orthogonal, and note that a")
        print("512-member HD block is an exactly orthonormal basis of R^512 while the iid")
        print("block has off-diagonal Gram entries up to ~0.2. The designs are maximally")
        print("different and the estimator cannot tell them apart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
