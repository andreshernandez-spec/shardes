# E13 GRPO reference arm, A100, 2026-08-17

One GRPO run on Countdown with Qwen2.5-0.5B-Instruct at Qiu et al.'s published
settings, `grpo.yaml` at commit 81c9aa8: lr 1e-5 held constant (GRPO-Zero has no
scheduler), kl beta 1e-3, group 30, 8 prompts per optimizer step, temperature 1.0
with top-p/top-k off (verified against GRPO-Zero's sampler), 300 steps, seed 0.
Implementation is TRL 1.10.0; `env.txt` and `pip-freeze.txt` record the stack,
`trainer_state.json` is the full per-step log, and `summarize.py` reproduces the
numbers below. One A100-SXM4-80GB, ~30 min wall clock.

| steps | mean training reward |
|---|---|
| 1 | 0.022 |
| 1-20 | 0.117 |
| 50-69 | 0.127 |
| 100-119 | 0.123 |
| 200-219 | 0.128 |
| 280-299 | 0.134 |

The shape: reward jumps to the 0.1 format tier inside twenty steps, then stays
within noise of it for the remaining 280. By the end the policy is nearly
deterministic (entropy 0.011) and every prompt's generation group scores
identically (frac_reward_zero_std 1.00), at which point the group-relative
advantage is zero and GRPO has no learning signal left. KL settles at 0.35.

Read against the ES arms (results/pilot-a100-2026-08-16, last-50 reward 0.166
to 0.172 with ~8% solves and still climbing): suggestive, not conclusive. One
seed per arm, this column is training reward on temperature-1 samples while the
ES numbers are greedy decode, and GRPO stopped at 72,000 sample evaluations
against the ES arms' 120,000 (300 x 240 vs 500 x 240; Qiu's protocol matches
totals, so a matched comparison wants 500 GRPO steps or a cut ES curve). The
held-out 2000-puzzle eval in the E13 design is where any claim gets settled.
