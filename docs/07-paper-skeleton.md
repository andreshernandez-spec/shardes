# Paper skeleton — scaffolding, not prose

Every sentence of the paper is Andres's to write (CLAUDE.md ground rule 1). What this
file holds is the load-bearing structure underneath: per section, the claim it must
carry, the evidence that carries it, and the number-to-source map so no figure or
statistic enters the text without its committed script. Sections follow docs/05's
outline; this file adds what exists now that the measurements are done.

Status 2026-08-19: every figure and table exists except F3 (needs E5/TRC) and F6
(superseded: F7 carries the end-to-end story with better controls). E9's TPU side is
the one open measurement.

## 1. Introduction

Carries: two 2025 papers made ES work at LLM scale with opposite structural bets
(Qiu: full-rank + seed regeneration; EGGROLL: rank-r factored), no common library can
express both, and the folk claim "ES only communicates scalars" was never measured.
Closes with the contributions list, each pointing at its section.

- The evosax foreclosure (ravel_pytree) is a checkable code fact: docs/00.
- The folk-claim tension resolves in §6: true for strategy A in order of magnitude
  (E0/TB2), and strategy B — the one that all-reduces a model-sized buffer — WINS over
  much of the grid anyway (F2), which is the more interesting sentence.

## 2. Background

ES estimator, the two algorithms, why one library forecloses both. Mostly citations
plus docs/00's framing. No numbers of ours; nothing to source.

## 3. Design

sample / apply / contract, the seed contract, replicated distribution state, the
shaping barrier as one deliberate line. Sources are code and docs/02. The three
design decisions worth prose: member-indexed seeds (device-count invariance,
invariant 2), never materializing structured perturbations (invariant 3), and the
unconditional replicate in `tell` (E10 priced it: gather 2.4-6 us at every measured
N and D; the sort is the cost and it is local).

## 4. The contraction question

The analytic cost model, written before E4 ran (docs/03). State that order honestly;
it is the paper's methodological teeth. The prediction to check against F2: B beats A
where the model all-reduce amortizes over enough member-work; A returns where
perturbation work per member is small (low rank) and d large.

## 5. Experimental setup

Platforms (8x A100 SXM4-80GB; TPU v5e-8, 16 GB/chip), the matched-shapes rule
(docs/05 E8 note), measurement protocol (3 warmup, 5 repeats, median+IQR,
block_until_ready, matmul precision pinned and recorded per record), and the
provenance rule: every record stamps commit + env, configs committed before runs.
The bf16 fitness refusal and the trajectory guard get a paragraph here.

## 6. Results — the number-to-source map

| claim in text | evidence | source |
|---|---|---|
| scaling to 8 devices, both platforms | F1 | `phase2/figures/f1-scaling.png`, `plot_paper.py` |
| crossover exists, moves with platform | F2 | `f2-crossover.png`; sign flip inside low-rank panels |
| dense ES flatlines on v5e weak scaling | F1 lower-right | `results-tpu-v5e8` |
| low-rank vs dense cost; feasibility staircase | F4 | `f4-cost-*.png`, `plot_cost.py` |
| no TPU inversion of EGGROLL's motivation (C4) | F4 rows | `results-cost`, `results-cost-tpu-v5e8` READMEs |
| barrier: gather 2.4-6 us, sort 12.1 ms at 2^18, D-independent | §6 text or small table | `results-barrier-tpu-v5e8/README.md`, `barrier.py` |
| baseline throughput parity | TB1 | `results-m4-*` (GPU; TPU side open, E9) |
| comm accounting, analytic vs measured | TB2 | `comms*.json`, docs/03 M5 |
| ablations | TB3 | `tb3.py` output, verbatim |
| highest-vs-default costs 1.6-1.9x only for low-rank arms | TB3 row | `results-cost-tpu-v5e8-highest/README.md` |

## 7. End-to-end (C6, replaces the conditional coupling section)

F7 plus the C6a-d sub-claims: rank axis flat on held-out Countdown at N=30 (and F5's
curves predict that tie at this N, C6b); frozen embedding free (C6c); bitwise
determinism, D-invariance, and the program-boundary caveat (C6d). GRPO reference with
Qiu's settings, untuned, 2 of 3 seeds degenerate — report as variance, not victory.
The cross-decoder delta (0.054 vs 0.037 base) bounds cross-arm claims; within-family
comparisons share one decoder. Sources: `countdown/results/*/README.md`, `plot_e13.py`.

## 8. Limitations

docs/05's list, plus earned ones: E4 at sweep resolution (denser grid is TRC work);
single-node GPU; v5e 16 GB/chip means absolute N doesn't match H100 figures; the
r=1 pad (a TPU-compiler workaround measured to be numerically invisible, PR #54);
the barrier numbers are isolation ceilings, read next to in-context times.

## 9. Related work

docs/00 has the list. Nothing to source.

## Open decisions for Andres

1. ~~Venue/length target~~ **Decided 2026-08-20: MLSys-style full paper.** That
   resolves the mechanical follow-ons: M1-M3 and the per-platform figure sets go to
   the appendix, TB1-TB3 stay in the main body, section 7 is a full section, and TB1
   waits for E9-TPU (the timeline allows one more quota week). Likely deadline is
   late October 2026 by the 2026 cadence; confirm when the CFP lands.
2. ~~F1's two flagged judgement calls~~ **Decided 2026-08-20: as drafted.** The
   largest common cell (d=2048), absolute time + throughput. The opener's job is
   "runs fast on real hardware at real sizes"; efficiency-vs-ideal lives in M1 in
   the appendix, and the TPU's missing D=1 point (the dense OOM) stays visible with
   one caption sentence rather than a normalization convention.
3. Whether section 7 leads with F7 or with the determinism story.
