#!/usr/bin/env python
"""C6d verdicts from the five demo runs. Run where their results dirs live.

Three separable claims, in order of strength:
1. Determinism: the two D=8, 20-generation runs from one seed must agree byte for
   byte in their logs and exactly in their final parameters.
2. Single-update device invariance at full scale: D=1 vs D=8 after one generation
   from the same init, parameters within invariant 2's tolerance (the update is
   f32-accumulated from bf16 noise; the D axis only reassociates the sum).
3. Trajectory context, not pass/fail: D=1 vs D=8 over 20 generations. Greedy
   decode amplifies the tolerated per-update difference (one near-tie argmax flip
   anywhere diverges the texts), so agreement here is a bonus and divergence is
   expected behavior, reported with the generation where rewards first differ.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RTOL = 1e-5


def params(d):
    z = np.load(HERE / d / "state.npz", allow_pickle=False)
    return [z[k] for k in sorted(z.files) if k.startswith("p")]


def rewards(d):
    return [json.loads(l)["reward_mean"] for l in (HERE / d / "log.jsonl").open()]


def max_rel(a, b):
    worst = 0.0
    for x, y in zip(a, b):
        x, y = x.astype(np.float64), y.astype(np.float64)
        denom = np.maximum(np.abs(y), 1e-30)
        worst = max(worst, float(np.max(np.abs(x - y) / denom)))
    return worst


# 1. determinism
det = max_rel(params("results-c6d-d8-g20"), params("results-c6d-d8-g20-repeat"))
r1, r2 = rewards("results-c6d-d8-g20"), rewards("results-c6d-d8-g20-repeat")
print(f"determinism (D=8 twice, 20 gens): max param rel err {det:.2e}, "
      f"rewards {'identical' if r1 == r2 else 'DIFFER'} -> "
      f"{'PASS' if det == 0.0 and r1 == r2 else 'FAIL'}")

# 2. single-update invariance
inv = max_rel(params("results-c6d-d1-g1"), params("results-c6d-d8-g1"))
print(f"one-update invariance (D=1 vs D=8): max param rel err {inv:.2e} "
      f"(tolerance {RTOL:.0e}) -> {'PASS' if inv <= RTOL else 'FAIL'}")

# 3. trajectory context
ra, rb = rewards("results-c6d-d1-g20"), rewards("results-c6d-d8-g20")
first = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), None)
tail = max_rel(params("results-c6d-d1-g20"), params("results-c6d-d8-g20"))
if first is None:
    print(f"trajectory (20 gens): rewards identical throughout, "
          f"final max param rel err {tail:.2e}")
else:
    print(f"trajectory (20 gens): rewards first differ at generation {first} "
          f"(expected: greedy decode amplifies the tolerated per-update "
          f"difference), final max param rel err {tail:.2e}")
