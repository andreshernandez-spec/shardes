# E13 pilot, ES arms, A100, 2026-08-16

Four ES arms on Countdown with Qwen2.5-0.5B-Instruct, 500 generations each,
configs `pilot.yaml` and `pilot-lr{1,4,16}.yaml` at commit 4eadcd2 (which each
log records in its generation-0 `env` line, along with jax 0.11.0). One RunPod
community pod, NVIDIA A100-SXM4-80GB, one run per arm, seed 0.

Run `summarize.py` in this directory to reproduce the table.

| arm | median s/gen | total min | reward | solved | format |
|---|---|---|---|---|---|
| mirrored-seed (full rank, Qiu) | 4.5 | 57.2 | 0.168 | 0.076 | 1.00 |
| lr1 | 2.0 | 29.1 | 0.172 | 0.081 | 0.99 |
| lr4 | 3.4 | 60.6 | 0.167 | 0.075 | 1.00 |
| lr16 | 7.2 | 284.5 | 0.166 | 0.073 | 1.00 |

reward, solved, format are means over the last 50 generations. All four arms
learn: reward from ~0.03 to ~0.17, solve rate from ~1% to ~8%, and every arm
reaches full format compliance. On this single seed the four noise structures
are indistinguishable on final reward; rank-1 gets there in half the wall
clock of full rank (2.0 vs 4.5 s/gen steady state). That is the C6a question
(docs/05) answered in the direction the rank axis hoped for, but one seed
decides nothing; N seeds and the held-out eval are the E13 proper.

Two operational notes, both with evidence here:

- `gate-probe-failed-2026-08-15.jsonl` is the first attempt, gated off at
  304.6 s/gen. The cause was the driver calling `tell` un-jitted, which
  recompiled its contract scan every generation (steady state measured at
  eval 4.1 s, tell ~314 s). Fixed in PR #30; `gate-probe.jsonl` is the same
  probe after the fix, passing at 4.3 s/gen.
- lr16's total is 4.7 h despite 7.2 s/gen steady state: generations 0 and 1
  spent 6622 s and 6845 s in XLA compile. Diagnosed after the run: tell
  regenerates the factors from seeds, and LowRank.sample unrolled a Python
  loop over the 2r column keys into that graph, so compile time scaled with
  rank. Fixed by vmapping the coupling over the column keys (bit-identical
  noise); `experiments/lowrank_compile_diag.py` has the before/after numbers.
