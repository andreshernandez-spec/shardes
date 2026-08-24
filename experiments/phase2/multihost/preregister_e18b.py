#!/usr/bin/env python
"""E18/E18b: the sign map across inter-node bandwidth, frozen before the rental.

    python preregister_e18b.py            # markdown to stdout
    python preregister_e18b.py --write    # + results-e18b/preregistration.json

`predict.py` and `predict_e18b.py` run ON the cluster, after the preflight has measured
that fabric. This runs BEFORE it, from committed single-node data only, and writes down
what the model says for a range of fabrics the rental might turn out to have. Its whole
purpose is to exist in git history with a timestamp earlier than the session log, so H2,
H3, G2 and G3 are judged against a prediction and not against a recollection.

Inputs, and nothing else:
  - `results-consistent`, the committed 8x A100 D=8 sweep: delta_8 = t_B - t_A per cell;
  - `results-ladder/ladder-nvidia-a100-sxm4-80gb-D8.json`, NVLink's in-program all-reduce
    at 4P, which is the intra-node term the boundary replaces;
  - C per cell, solved from those two (`docs/11-cost-model.md`), because
    `contraction_isolation.py` has not run on an A100 yet. The preflight runs it and
    `predict.py` then uses the measured value; where the two disagree, the measured one
    wins and this file is the record of what changed.

    delta_16(beta) = delta_8 + [alpha_inter + 4P/beta - ar_nvlink(4P)] - C/16
    flip at        beta* = 4P / (-delta_8 + ar_nvlink + C/16 - alpha_inter)

THREE THINGS THIS PREDICTION CAN GET WRONG, stated here rather than after the fact:

1. C is solved, not measured. On the low-rank cells it solves NEGATIVE (the open term in
   `docs/11`), and is clamped to zero here. Clamping understates B's compute saving, so
   it favors A; those cells are predicted A-wins by 5 to 800 ms, far outside the clamp,
   and the sign is safe. On the seed cells C is large and positive and the C/16 term is
   1.6 to 3.9 ms against boundary penalties of tens to hundreds, so the sign is not
   sensitive to it either.
2. The nominal rates are line rates. A ring all-reduce puts roughly 2(D-1)/D times the
   payload on the wire, and beta fitted from a measured all-reduce absorbs that while a
   nominal 10gbit does not. The throttled predictions can therefore be optimistic by up
   to about 2x, which moves each flip bandwidth by the same factor and leaves the
   ORDERING of the three flip points alone.
3. alpha_inter is a placeholder until the preflight measures it. At 4P = 96 MiB the
   bandwidth term is 4 ms even on 25 GB/s IB, so anything from 50 to 500 us changes no
   sign in the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import costmodel

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "results-consistent"
LADDER = HERE.parent / "results-ladder" / "ladder-nvidia-a100-sxm4-80gb-D8.json"
OUT = HERE / "results-e18b"

#: placeholder until the preflight; see caveat 3.
ALPHA_INTER = 100e-6
DEVICES = 16

#: The E18 and E18b cells (e18.yaml, e18b.yaml), both arms.
CELLS = [(arm, d, n) for arm in ("seed_regenerated", "mirrored_lr1")
         for d, n in ((512, 1024), (2048, 128), (2048, 256))]

#: The bracket of fabrics a rented 2-node cluster might have, plus the throttle settings.
FABRICS = [("1gbit", 0.125e9), ("10gbit", 1.25e9), ("socket 2 GB/s", 2.0e9),
           ("25 GbE", 3.125e9), ("100 GbE", 12.5e9), ("200Gb IB", 25.0e9)]


def anchor(arm: str, d: int, n: int) -> float | None:
    """delta_8 = t_B - t_A from the committed single-node sweep."""
    fa = SWEEP / f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=A.json"
    fb = SWEEP / f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=B.json"
    if not (fa.exists() and fb.exists()):
        return None
    return (json.loads(fb.read_text())["seconds_median"]
            - json.loads(fa.read_text())["seconds_median"])


def nvlink_allreduce(nbytes: int) -> float:
    """The in-program step cost at this payload, interpolated between ladder points."""
    rec = json.loads(LADDER.read_text())
    pts = sorted((int(k), v["step_seconds"]) for k, v in rec["allreduce"].items())
    if nbytes <= pts[0][0]:
        return pts[0][1]
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if nbytes <= b1:
            return t0 + (t1 - t0) * (nbytes - b0) / (b1 - b0)
    return pts[-1][1]


def build() -> dict:
    out = {"alpha_inter_seconds": ALPHA_INTER, "devices": DEVICES,
           "fabrics": {n: b for n, b in FABRICS}, "cells": {}}
    for arm, d, n in CELLS:
        delta_8 = anchor(arm, d, n)
        if delta_8 is None:
            continue
        p_bytes = costmodel.params_bytes(d)
        ar_nv = nvlink_allreduce(p_bytes)
        ag = nvlink_allreduce(4 * n)
        # C solved from the anchor, clamped at zero: caveat 1.
        c_solved = (-delta_8 - ag + ar_nv) * 8 / 7
        c = max(c_solved, 0.0)
        need = -delta_8 + ar_nv + c / DEVICES - ALPHA_INTER
        out["cells"][f"{arm}__d={d}__N={n}"] = {
            "delta_8_seconds": delta_8,
            "contraction_solved_seconds": c_solved,
            "contraction_used_seconds": c,
            "contraction_clamped": c_solved < 0,
            "nvlink_allreduce_seconds": ar_nv,
            "flip_beta_bytes_per_second": (p_bytes / need) if need > 0 else None,
            "delta_16_by_fabric": {
                name: delta_8 + (ALPHA_INTER + p_bytes / beta - ar_nv) - c / DEVICES
                for name, beta in FABRICS},
        }
    return out


def markdown(rec: dict) -> str:
    names = [n for n, _ in FABRICS]
    lines = ["predicted `delta_16 = t_B - t_A` in ms; negative = B wins, positive = A wins",
             "",
             "| cell | delta_8 | C used | flip beta | " + " | ".join(names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for key, v in rec["cells"].items():
        flip = v["flip_beta_bytes_per_second"]
        flip_s = "no flip" if flip is None else f"{flip / 1e9:.2f} GB/s"
        star = "*" if v["contraction_clamped"] else ""
        row = ("| %s | %+.2f | %.2f%s | %s |"
               % (key.replace("__", " "), v["delta_8_seconds"] * 1e3,
                  v["contraction_used_seconds"] * 1e3, star, flip_s))
        row += " " + " | ".join("%+.2f" % (v["delta_16_by_fabric"][n] * 1e3)
                                for n in names) + " |"
        lines.append(row)
    lines += ["", "`*` C solved negative and clamped to zero; see caveat 1.",
              f"alpha_inter placeholder {rec['alpha_inter_seconds'] * 1e6:.0f} us, D={rec['devices']}."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rec = build()
    print(markdown(rec))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "preregistration.json").write_text(json.dumps(rec, indent=1))
        print(f"\nwrote {OUT / 'preregistration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
