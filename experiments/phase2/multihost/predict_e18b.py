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
what makes G4 a prediction. The model reuses predict.py's coarse form (a sign
and a bracket), extended so beta is the setting's bandwidth:

    comm_bump(setting) = ladder(alpha_socket, beta_setting, 4P) - ladder_1x8(4P)
    delta_2x8(setting) in [delta_8 + bump - |delta_8|, delta_8 + bump]

Latency is held at the socket-native alpha: tbf caps bandwidth, not latency.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

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
                p_bytes = 4 * 6 * d * d
                bump = ladder(alpha_s, beta, p_bytes) - ladder(a1x8, b1x8, p_bytes)
                lo, hi = delta_8 + bump - abs(delta_8), delta_8 + bump
                mid = (lo + hi) / 2
                predictions[f"{name}__{arm}__d={d}__N={n}"] = {
                    "beta_bytes_per_second": beta,
                    "beta_nominal": name != "socket-native",
                    "delta_1x8_measured": delta_8,
                    "comm_bump_predicted": bump,
                    "delta_2x8_bracket": [lo, hi],
                    "predicted_sign_B_minus_A": "B_wins" if mid < 0 else "A_wins",
                }
                print(f"{name} {arm} d={d} N={n}: beta {beta/2**30:.2f} GiB/s, "
                      f"bump {bump*1e3:+.2f} ms -> sign "
                      f"{predictions[f'{name}__{arm}__d={d}__N={n}']['predicted_sign_B_minus_A']}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "predictions-e18b.json").write_text(json.dumps(predictions, indent=1))
    print(f"wrote {RES / 'predictions-e18b.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
