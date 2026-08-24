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
    contraction  = B's contraction work splits further from D=8 to D=16
                   while A's replicated contraction stays constant. That
                   gain is C (1/8 - 1/16), and `contraction_isolation.py`
                   measures C. Without a record for the cell this falls back
                   to the coarse bracket [0, |delta(1x8)|] the file used
                   before C existed, and every prediction says which it used.

    predicted delta(2x8) = delta(1x8) + comm_bump - C (1/8 - 1/16)

The SIGN prediction is what H2 and H3 are judged on; the magnitude is
recorded either way so a miss is visible and honest. The 2x4 topology needs
no prediction: H1 IS its prediction (t_A unchanged, t_B up by the same
comm_bump at D=8's contraction split, with no C term at all).

This file must be run, and predictions.json written, before any 2x8 ES
cell; the session log's timestamps are the witness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import costmodel

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
            p_bytes = costmodel.params_bytes(d)
            bump = (ladder_seconds(pre_2x8, p_bytes)
                    - ladder_seconds(pre_1x8, p_bytes))
            c = costmodel.measured_contraction(arm, d, n)
            pred = costmodel.predict_delta(delta_8, bump, 16, c)
            predictions[f"{arm}__d={d}__N={n}"] = {
                "delta_1x8_measured": delta_8,
                "comm_bump_predicted": bump,
                "delta_2x8_bracket": pred["delta_bracket"],
                **pred,
            }
            lo, hi = pred["delta_bracket"]
            span = (f"{pred['delta_predicted'] * 1e3:+.2f} ms"
                    if pred["contraction_source"] == "measured"
                    else f"[{lo * 1e3:+.2f}, {hi * 1e3:+.2f}] ms")
            print(f"{arm} d={d} N={n}: delta(1x8) {delta_8 * 1e3:+.2f} ms, "
                  f"bump {bump * 1e3:+.2f} ms, C {pred['contraction_source']} "
                  f"-> {span}, sign: {pred['predicted_sign_B_minus_A']}")

    out = RESULTS / "predictions.json"
    out.write_text(json.dumps(predictions, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
