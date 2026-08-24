#!/usr/bin/env python
"""E18b G4: predict t_B - t_A at each throttle setting before its cells run.

    python predict_e18b.py       # writes results-e18b/predictions-e18b.json

Calibration inputs, and nothing else:
  - the committed single-node D=8 sweep (results-consistent): delta_8 = t_B - t_A
    per (arm, cell), the same anchor predict.py uses;
  - results-e18/preflight-1x8.json: the intra-node (NVLink) ladder;
  - results-e18b/preflight-2x8__set=socket-native.json: the socket transport's
    alpha and beta, measured unthrottled.

The throttled points (10gbit, 1gbit) are predicted from their NOMINAL rate, not
from their own ladders, which by design do not exist yet when this runs: that is
what makes G4 a prediction. The model is `costmodel.py`, shared with predict.py,
with beta set by the setting:

    comm_bump(setting) = ladder(alpha_socket, beta_setting, 4P) - ladder_1x8(4P)
    delta_2x8(setting) = delta_8 + bump - C (1/8 - 1/16)

Latency is held at the socket-native alpha: tbf caps bandwidth, not latency.

Where `contraction_isolation.py` has measured C for the cell this is a point
prediction; without it, the coarse [0, |delta_8|] bracket, flagged as such. G2
asks WHERE the seed_regenerated sign flips, and that needs C: the flip is at

    beta* = 4P / (ag(4N) + C (D-1)/D - alpha_socket)

which this writes out per cell, in GiB/s, beside the three settings that will
be run. A predicted flip bandwidth that falls between two settings is the
sharpest form of G2 the sweep can test.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import costmodel

HERE = Path(__file__).resolve().parent
RES_E18 = HERE / "results-e18"
RES = HERE / "results-e18b"
SWEEP = HERE.parent / "results-consistent"

# nominal bandwidths, bytes/sec; socket-native uses its measured beta instead
RATE_BYTES = {"rate-10gbit": 10e9 / 8, "rate-1gbit": 1e9 / 8}


def ladder(alpha: float, beta: float, nbytes: float) -> float:
    return alpha + nbytes / beta


def main() -> int:
    cfg = yaml.safe_load((HERE / "e18b.yaml").read_text())
    pre_1x8 = json.loads((RES_E18 / "preflight-1x8.json").read_text())
    sn = json.loads((RES / "preflight-2x8__set=socket-native.json").read_text())
    alpha_s = sn["alpha_seconds"]
    beta_by_setting = {"socket-native": sn["beta_bytes_per_second"], **RATE_BYTES}

    a1x8 = pre_1x8["alpha_seconds"]
    b1x8 = pre_1x8["beta_bytes_per_second"]

    predictions: dict = {}
    for s in cfg["settings"]:
        name = s["name"]
        beta = beta_by_setting[name]
        for arm in cfg["arms"]:
            for cell in cfg["cells"]:
                d, n = cell["d"], cell["N"]
                ka = SWEEP / f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=A.json"
                kb = SWEEP / f"mode=strong__D=8__d={d}__N={n}__s={arm}__how=B.json"
                if not (ka.exists() and kb.exists()):
                    print(f"no D=8 anchor for {arm} d={d} N={n}; skipped")
                    continue
                delta_8 = (json.loads(kb.read_text())["seconds_median"]
                           - json.loads(ka.read_text())["seconds_median"])
                p_bytes = costmodel.params_bytes(d)
                bump = ladder(alpha_s, beta, p_bytes) - ladder(a1x8, b1x8, p_bytes)
                c = costmodel.measured_contraction(arm, d, n)
                pred = costmodel.predict_delta(delta_8, bump, 16, c)
                # The fitness gather is latency at these populations on every fabric
                # measured so far, so alpha stands in for ag(4N) in the flip solve.
                flip = costmodel.flip_bandwidth(alpha_s, alpha_s, d, 16, c)
                predictions[f"{name}__{arm}__d={d}__N={n}"] = {
                    "beta_bytes_per_second": beta,
                    "beta_nominal": name != "socket-native",
                    "delta_1x8_measured": delta_8,
                    "comm_bump_predicted": bump,
                    "delta_2x8_bracket": pred["delta_bracket"],
                    "flip_beta_bytes_per_second": flip,
                    **pred,
                }
                lo, hi = pred["delta_bracket"]
                span = (f"{pred['delta_predicted'] * 1e3:+.2f} ms"
                        if pred["contraction_source"] == "measured"
                        else f"[{lo * 1e3:+.2f}, {hi * 1e3:+.2f}] ms")
                at = "" if flip is None else f", flips at {flip / 2**30:.2f} GiB/s"
                print(f"{name} {arm} d={d} N={n}: beta {beta/2**30:.2f} GiB/s, "
                      f"bump {bump*1e3:+.2f} ms, C {pred['contraction_source']} -> "
                      f"{span}, sign {pred['predicted_sign_B_minus_A']}{at}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "predictions-e18b.json").write_text(json.dumps(predictions, indent=1))
    print(f"wrote {RES / 'predictions-e18b.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
