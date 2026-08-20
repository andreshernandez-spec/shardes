# E14: stability under hyperparameter perturbation (claim C7)

Designed 2026-08-20, at Andres's direction: not a citation about RL fragility, an
experiment. Configs are committed before any run and cited by SHA; the multiplier-1
columns come from E13's committed curves, so E14 extends the campaign rather than
starting a new one.

## The claim being tested

**C7: at matched task, model and evaluation budget, ES tolerates order-of-magnitude
perturbation of its published hyperparameters with graceful degradation, while
critic-free GRPO-style RLVR's viable region is narrow and its failures are
catastrophic (collapse below the base model), even with its stabilization machinery
active; removing its KL anchor alone breaks it at its own published settings, and ES
has no analogous component to remove.**

Scoped to critic-free GRPO-style RLVR deliberately (2026-08-20, from the research
spike, docs/09): vanilla PPO WITH a learned critic trains stably at 32B with no KL
penalty at all (Open-Reasoner-Zero, arXiv:2503.24290), so "RL is fragile" is not a
defensible claim; "the critic-free RLVR recipe our baseline uses is fragile" is,
and it is the recipe the 2025 reasoning wave actually runs.

The mechanism is stated, not hidden: rank-based fitness shaping makes the ES update
invariant to reward scale, and the step size is sigma-relative by construction. That
invariance IS the stability being measured. On the RL side, the machinery whose
necessity is being tested is exactly what the field added to make updates slow and
gradual: the KL anchor to a frozen reference (kl_beta), the clipped ratio, the tuned
learning rate.

## Arms

All runs: Countdown, Qwen2.5-0.5B-Instruct, same puzzle pool, same 2000-puzzle
held-out greedy eval as E13, horizon 150 generations/steps (E13's failures appear
by step 50; 150 gives three eval points past onset), 3 seeds (0, 1, 2).

| arm | dial | values | runs |
|---|---|---|---|
| ES lr | lr multiplier | 1/8, **1**, 8 | 2x3 new (x1 from E13) |
| ES sigma | sigma multiplier | 1/4, **1**, 4 | 2x3 new (x1 shared) |
| GRPO lr | lr multiplier | 1/8, **1**, 8 | 2x3 new (x1 from E13) |
| GRPO anchor | kl_beta | **1e-3**, 0 | 1x3 new |
| GRPO clip | clip_epsilon | **0.2**, 10 (never binds) | 1x3 new |

Bold = published checkmarked value (Qiu Table 4; the clip is TRL's default, now
pinned explicitly in every generated config); multipliers are applied to each
algorithm's own published setting, which is the only fair definition of
perturbation. ES arm is mirrored rank 1 (the paper's headline strategy). 24 new
runs.

The clip arm (added 2026-08-20 after the research spike) is the second
single-stabilizer ablation: the entropy-collapse literature ties the collapse
mechanism to the clip specifically (arXiv:2505.22617, arXiv:2509.26114), while
Engstrom et al. found PPO-NoClip can match PPO once code-level tricks are present.
It is informative whichever way it lands, and it is run BECAUSE it might not
confirm: an ablation that cannot fail is not evidence.

## Metrics, defined before the data

Per run, from eval.jsonl (evals at 0, 50, 100, 150):

- **final**: held-out reward at the last eval.
- **collapsed**: final < the arm's own base-model floor (each decoder family
  against its own generation-0 floor, per the E13 cross-decoder rule).
- **frozen**: (final - floor) < 0.25 x (median x1-arm final - floor): less than a
  quarter of the reference progress.
- **drawdown**: max over t of (max_{s<=t} R_s - R_t).
- **drift**: parameter L2 distance from the pretrained weights
  (`param_l2_from_init`), per eval for ES and at the final checkpoint for GRPO.
  Added because the two strongest counter-papers (arXiv:2604.01499,
  arXiv:2601.20861) predict ES buys reward stability with large off-task
  movement; measuring it ourselves turns their attack into a reported row.
- **entropy and completion length** (GRPO arms): run_grpo.py now dumps the full
  TRL log history (train-log.jsonl), so whatever this TRL version logs
  (entropy, completions/mean_length, kl) is preserved and extracted at
  analysis, connecting collapses to the covariance mechanism of
  arXiv:2505.22617 and separating length-bias pathologies (arXiv:2503.20783)
  from reward collapse. The exact key names are confirmed at pod smoke time;
  entropy_coef stays 0 (we log entropy, never regularize it).

Per arm: the robustness curve (F8): x = multiplier (log scale), y = final, one
panel per algorithm, seed markers, floor dashed, collapsed runs marked. The
headline statistic: count of collapsed runs per algorithm across all perturbed
cells.

**Counts, not variance.** Three seeds per cell cannot support variance estimates,
and Henderson et al. is the citation for why pretending otherwise is the field's
own bad habit. The claim rests on binary failure counts (collapsed / frozen)
across 24 perturbed runs plus E13's 6 published-settings runs, which 3 seeds per
cell legitimately supports.

## What would falsify C7

GRPO tolerating lr x8 without collapse or freeze; ES collapsing at any tested
multiplier; the kl_beta=0 arm training fine. The clip-off arm is exempt from
this list by design: Engstrom et al. make either outcome plausible, so it
informs rather than falsifies. Any of these gets reported as found;
the experiment is worth running precisely because E13 (2 of 3 GRPO seeds degenerate
at the published settings themselves) makes the outcome likely but not certain.

## Honest scope

One task, one model size, one RL algorithm, 150-step horizon, 3 seeds. The claim
licensed is about this setting, stated next to Henderson et al.'s general finding,
not as a theorem. GRPO keeps all its stabilizers in the lr arms; only the ablation
arm removes one, one at a time.

## Cost

ES short runs are ~15-20 min on one A100 (~$0.4 each, 12 runs); GRPO 150-step runs
~2 h (~$2.8 each, 12 runs). Estimate $40-60 total, one rented A100, sequential,
resumable per run. Configs in `experiments/countdown/e14/`, generated by
`e14_make_configs.py` (committed, so the derivation from the E13 bases is
reviewable), runs driven by the existing run_es.py / run_grpo.py with --seed.

## Paper placement

Section 7 gains a subsection after the seed-variance paragraph; F8 joins the
figure table; the introduction's motivation paragraph (RL's stabilization
machinery: target networks and Polyak averaging as EWMA'd weights, trust regions
and KL anchors as slow-shift constraints, hand-built curricula) cites this
experiment instead of only the literature.
