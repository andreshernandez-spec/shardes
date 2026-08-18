# E13 campaign: held-out curves, matched totals, three seeds. A100, 2026-08-17

Everything the two earlier result files hedged on, closed in one campaign: a
2000-puzzle held-out eval disjoint from the training pool by construction, GRPO
extended to the matched 120,000 sample evaluations, and seeds 0-2 for every arm.
Four ES arms (Qiu full rank via seed regeneration, EGGROLL-style rank 1/4/16)
and GRPO per seed, one A100-SXM4-80GB, ~13 h wall clock, code at 445ba74
(eval_grpo.py at 579581c, see the caveat), environment in `env.txt`. Per-arm
learning curves are the `*-eval.jsonl` files (ES also has per-generation
training logs); `summarize.py` reproduces both tables; `campaign-log.txt.gz` is
the pod's stdout, which is where GRPO's per-step training metrics live.

ES arms, final held-out eval over seeds 0-2, mean [min-max]:

| arm | eval reward | solved |
|---|---|---|
| mirrored-seed (full rank) | 0.157 [0.154-0.162] | 0.064 [0.059-0.069] |
| lr1 | 0.155 [0.152-0.158] | 0.061 [0.058-0.064] |
| lr4 | 0.155 [0.149-0.160] | 0.061 [0.054-0.067] |
| lr16 | 0.153 [0.151-0.156] | 0.059 [0.057-0.062] |

GRPO, final held-out eval per seed: 0.144 / 4.9% solved (seed 0), 0.144 / 4.9%
(seed 1), 0.100 / 0.0% (seed 2). Base model before any training: 0.054 / 1.2%
solved, 43% well-formed (ES decoder).

Three findings:

1. **The noise-structure axis is flat on held-out quality.** Twelve ES runs land
   in 0.149-0.162; the seed-to-seed spread inside any arm covers the spread
   between arms. Rank 1 buys full-rank quality at the cheapest decode (2.0
   s/generation steady against full rank's 4.5), which is C6a's cost-at-matched-
   quality claim with N=3 and a held-out metric.
2. **ES is stable where GRPO is not.** Every ES run learned (all curves reach
   ~99% format by generation 50 and climb to ~6% solves); GRPO froze at 4.9%
   solves on two seeds (identical eval output checkpoint after checkpoint, the
   entropy-collapse stall visible in its training logs) and on seed 2 never
   solved a held-out puzzle at all, 0.100 flat from step 50, format only. Same
   hyperparameters, Qiu's published ones, all three seeds.
3. **Most ES generalization is early.** Held-out reward moves 0.054 to ~0.15 in
   the first 50 generations and gains ~0.01 in the remaining 450; the training
   reward's steady climb past that overstates what transfers.

Decode caveat, and it is a finding in its own right: greedy is not one rule.
Qwen2.5 ships repetition_penalty 1.1 in generation_config.json and HF generate
applies it under do_sample=False, so the GRPO evals in this campaign run
eval_grpo.py at 579581c, which pins repetition_penalty=1.0 to make both arms
plain argmax. Seed 0's first GRPO run was evaluated before that fix; its
penalized eval is kept as `grpo-s0-eval-penalized.jsonl` for the record, and
`grpo-s0-eval.jsonl` is a full clean re-run. A residual decoder delta remains
(the same base weights score 0.054 under the JAX evaluator and 0.037 under the
torch one, right-versus-left padding numerics in bf16), so cross-arm gaps
smaller than ~0.02 should not be leaned on; the ES-vs-ES comparison shares one
decoder and has no such term.

Costs: the full campaign, including the seed-0 GRPO re-run, was ~15 pod-hours,
~$21.
