#!/usr/bin/env python
"""E18 H4: predict the 2x8 cells before they run, from calibration only.

    python predict.py            # writes results-e18/predictions.json

Inputs, and nothing else:
  - results-e18/preflight-1x8.json and preflight-2x8.json (the psum ladders
    measured by preflight on THIS cluster, intra-node and cross-node), and
  - the committed single-node A100 sweep (results-consistent), which this
    cluster's own 1x8 topology re-measures as its anchor.

The model is deliberately coarse and stated as such: a sign-and-bracket
prediction, not a fit.

For each (arm, d, N) cell, with P = 6 d^2 parameters:

    delta(1x8)   = t_B - t_A, measured, committed single-node sweep at D=8.
    comm_bump    = ar_2x8(4P) - ar_1x8(4P), the all-reduce's fabric penalty,
                   interpolated from each topology's psum ladder (alpha +
                   bytes/beta).
    contraction  = B's contraction work halves again from D=8 to D=16 while
                   A's replicated contraction stays constant; the gain is
                   bounded between 0 and half of B's D=8 contraction share,
                   which we bracket by [0, |delta(1x8)|] rather than
                   estimate: it cannot exceed the whole measured gap.

    predicted delta(2x8) in [delta(1x8) + comm_bump - bracket,
                             delta(1x8) + comm_bump]

The SIGN prediction is the midpoint's sign; H2 and H3 are judged on signs,
and the bracket is printed so a magnitude miss is visible and honest. The
2x4 topology needs no prediction: H1 IS its prediction (t_A unchanged, t_B
up by the same comm_bump at D=8's contraction split).

This file must be run, and predictions.json written, before any 2x8 ES
cell; the session log's timestamps are the witness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-e18"
SWEEP = HERE.parent / "results-consistent"


def ladder_seconds(pre: dict, nbytes: float) -> float:
    return pre["alpha_seconds"] + nbytes / pre["beta_bytes_per_second"]


def main() -> int:
    pre_1x8 = json.loads((RESULTS / "preflight-1x8.json").read_text())
    pre_2x8 = json.loads((RESULTS / "preflight-2x8.json").read_text())
    cfg = __import__("yaml").safe_load((HERE / "e18.yaml").read_text())

    predictions = {}
    for arm in cfg["arms"]:
        for cell in cfg["cells"]:
            d, n = cell["d"], cell["N"]
            key_a = f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=A.json"
            key_b = f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=B.json"
            fa, fb = SWEEP / key_a, SWEEP / key_b
            if not (fa.exists() and fb.exists()):
                print(f"no committed D=8 anchor for {arm} d={d} N={n}; skipped")
                continue
            ta = json.loads(fa.read_text())["seconds_median"]
            tb = json.loads(fb.read_text())["seconds_median"]
            delta_8 = tb - ta
            p_bytes = 4 * 6 * d * d
            bump = (ladder_seconds(pre_2x8, p_bytes)
                    - ladder_seconds(pre_1x8, p_bytes))
            lo = delta_8 + bump - abs(delta_8)
            hi = delta_8 + bump
            mid = (lo + hi) / 2
            predictions[f"{arm}__d={d}__N={n}"] = {
                "delta_1x8_measured": delta_8,
                "comm_bump_predicted": bump,
                "delta_2x8_bracket": [lo, hi],
                "predicted_sign_B_minus_A": "B_wins" if mid < 0 else "A_wins",
            }
            print(f"{arm} d={d} N={n}: delta(1x8) {delta_8 * 1e3:+.2f} ms, "
                  f"bump {bump * 1e3:+.2f} ms -> "
                  f"[{lo * 1e3:+.2f}, {hi * 1e3:+.2f}] ms, sign: "
                  f"{predictions[f'{arm}__d={d}__N={n}']['predicted_sign_B_minus_A']}")

    out = RESULTS / "predictions.json"
    out.write_text(json.dumps(predictions, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
