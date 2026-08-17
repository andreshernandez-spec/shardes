#!/usr/bin/env python
"""Reproduce the README numbers from trainer_state.json in this directory."""
import json
from pathlib import Path

h = json.load((Path(__file__).resolve().parent / "trainer_state.json").open())["log_history"]
rew = [(x["step"], x["reward"]) for x in h if "reward" in x]
print(f"steps logged: {len(rew)}; step 1 reward: {rew[0][1]:.3f}")
for lo in (1, 50, 100, 200, 280):
    w = [r for s, r in rew if lo <= s < lo + 20]
    print(f"steps {lo}-{lo + 19}: mean reward {sum(w) / len(w):.3f}")
kl = [x["kl"] for x in h if "kl" in x]
ent = [x["entropy"] for x in h if "entropy" in x]
zstd = [x["frac_reward_zero_std"] for x in h if "frac_reward_zero_std" in x]
print(f"final kl {kl[-1]:.2f}, final entropy {ent[-1]:.3f}, "
      f"final frac_reward_zero_std {zstd[-1]:.2f}")
