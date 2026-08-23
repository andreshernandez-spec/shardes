# 05 — The paper

What the paper claims, which experiment establishes each claim, and what the figures are.
The *how* — Kaggle sessions, TRC application timing, GCP mechanics — is in
`docs/06-benchmark-runbook.md`.

---

## The reframe that makes this affordable

Kaggle publishes a **TPU v5e-8: eight chips, free**, ~20 TPU-hours/week in 9-hour sessions.
That is the same device count as the $19.20/hr GCP 8×A100 node that `docs/compute.md`
originally built the plan around.

So the primary scaling study runs on TPU at **zero cost**, and the paid GPU session stops
being the main event and becomes the **cross-platform comparison** — which is both cheaper
and a better paper. TRC then extends the curve past 8 devices, also free, and its stated
obligation is literally to publish, which is what we're doing anyway.

---

## Claims

### C1 — The "ES only all-reduces scalars" folk claim is conditional, and we draw the boundary

There are two ways to close the ES update loop across devices:

- **Strategy A**: all-reduce `N` fitness scalars, then every device regenerates all `N`
  perturbations from seeds and contracts locally. Communication `O(N)`. Contraction
  replicated `D` times.
- **Strategy B**: each device contracts its local shard into a params-shaped partial, then
  `psum`. Communication `O(d)`, same as data-parallel SGD. Contraction split `D` ways.

The widely-repeated claim that ES needs only a scalar all-reduce is true for A and false
for B. Nobody has characterized where the crossover sits. We do, in `(N, d, D)`, on two
interconnects.

**This is the strongest single result in the paper** and the one that reads as distributed
systems rather than algorithm implementation.

**Phase 1 has the volume half, exactly.** `experiments/phase1/comms.py` reads every collective
out of the *optimized* HLO and puts it next to the analytic prediction, across 3 strategies ×
3 populations × 3 device counts. All 27 rows agree **to the byte**:

    A -> one all-gather,  8N bytes   (the shaped weights *and* the member ids)
    B -> one all-reduce,  4d bytes   (a tuple-valued psum over the whole params tree)

so A is independent of `D` and of the model, B is independent of `N`, and the crossover sits
at **`N = d/2`** — it moves with the *model size*, not with the device count. That is the
phase diagram in miniature, at one model size and in bytes.

Two things this found are worth carrying into the paper's methods section, because both are
mistakes the reader might otherwise repeat. The prediction was originally `4N` for A: it
gathers the member ids as well as the weights, since a device regenerating all `N`
perturbations has to know *which* members it is regenerating. And a first parser read only the
first array of each collective's result shape, which measured B at 64 B against 2112 because
XLA merges the per-leaf psums into one **tuple-valued** all-reduce. Prediction wrong once,
instrument wrong once; the table is only worth printing because it caught both.

**What Phase 1 cannot supply is the half that matters most.** Bytes are not time. Simulated
devices model volume faithfully and interconnect not at all, so where the crossover falls in
*wall-clock* — which is the actual claim — needs E4 and E7 on real hardware. Nothing above is
a timing result and none of it should be quoted as one.

**A second crossover, not previously anticipated.** `experiments/phase1/memory.py` measures
peak per-device scratch from the compiled executable. Per-device storage falls as `1/D`
exactly (128.53 MiB → 16.04 MiB across 8 devices, a factor of 8.0), but that is the least
interesting row: `SeedRegenerated` is **flat**, because its `contract` is a `lax.scan` that
never holds more than one member's noise, and `Mirrored(LowRank(r=1))` starts two orders of
magnitude lower and falls. So the two cross, between `D = 4` and `D = 8` at `N = 1024`.

The quotable form: **`SeedRegenerated` on one device uses 84× less than `IIDGaussian` on
eight.** Sharding divides; choosing the perturbation strategy changes the exponent. That
belongs in the paper next to C1's crossover, because it is the same shape of answer — which
strategy wins depends on the configuration, not on which paper one prefers.

### C2 — Both LLM-scale ES algorithms are expressible under one abstraction

Qiu et al. (full-rank, seed-regenerated, `N=30`) and EGGROLL (rank-`r` factored, never
materialized, `N` up to 2¹⁸) make opposite bets on the same tradeoff and have no common
library. `evosax` forecloses both by flattening solutions via `ravel_pytree`. We show a
perturbation-strategy abstraction under which the two are a two-line config diff, with no
global flattening and no loss of throughput versus each paper's own implementation.

Artifact claim. Established by implementation + baselines, not by a curve.

**The implementation half is done (Phase 1, 2026-07-31).** Both algorithms are one constructor
argument apart and both drive a real MuJoCo Playground environment end to end through the same
`init`/`ask`/`apply`/`tell`:

```python
ShardedES(strategy=Mirrored(SeedRegenerated()), n=30,      ...)   # Qiu et al.
ShardedES(strategy=Mirrored(LowRank(r=1)),      n=262_144, ...)   # EGGROLL
```

Device-count invariance holds on 1/2/4/8 simulated devices for both contraction strategies and
all three strategy families, and the portability half is confirmed on a real GPU. The
`ravel_pytree` ban is a static check. E0 is therefore substantially complete; what is left of
C2 is **E9, the baselines** — the "no loss of throughput versus each paper's own
implementation" clause is still unmeasured, and until it is, the claim is *expressible*, not
*competitive*. Do not let the two blur.

One constraint belongs in the paper rather than the appendix, because it is the first thing a
reader will hit: **models must route matmuls through `shardes.nn.dense`.** A structured weight
is substituted into the params tree, so a model doing arithmetic on it directly raises. That is
what buys low-rank perturbation without a jaxpr interpreter, and it is the honest cost of the
abstraction (`docs/01` C0.1).

### C3 — First systematic ES scaling study on TPU, and the interconnect matters

ES has an unusual communication profile: embarrassingly parallel rollouts, and either
`O(N)` scalars or one model-sized all-reduce per generation. TPU v5e's 2D torus ICI and
GPU NVLink have different collective cost structures, so the Strategy A/B crossover should
sit in a different place on each. Nobody has run ES at this scale on TPU at all.

### C4 — EGGROLL's low-rank motivation is GPU-shaped, and may invert on TPU

This is the claim I'd most want to be right about.

EGGROLL structures perturbations as rank-`r` **because naive ES is memory-bound on GPU** —
a batched matmul against `N` distinct weight matrices has terrible arithmetic intensity.
The fix is to never materialize, so all members share one base GEMM.

On TPU the MXU makes dense matmuls comparatively cheap while HBM capacity (16 GB/chip on
v5e) is the binding constraint. The cost balance that motivates the low-rank rewrite may
therefore invert, or shift substantially. Measuring `(N, m, n, r, dtype)` surfaces on both
platforms and showing where the rewrite pays is a real, unpublished result.

Structurally this is the same intellectual move as the FWHT crossover question — *where
does the fast transform beat the dense matmul, and why does the answer differ by
accelerator* — which is a coherent through-line if both projects get written up.

### C5 — Sample design does not improve ES gradient estimates in the `N/d_eff ≳ 1` regime — **settled, negative**

**Gate G0 answered "no" on 2026-07-30.** This is a resolved negative result and it is written
up as one, not as future work. Evidence: `experiments/phase0/`, 456 configurations at
`R = 30`, one uniform environment, 13.07 h on an RTX 3080. Full answer in
`docs/01-phase0-estimator-harness.md`.

The setup is what makes it worth a section. Low-rank perturbation genuinely inverts the `N/d`
arithmetic that makes sample design hopeless in high-dimensional ES: rank 1 samples in
`ℝ^(m+n)` rather than `ℝ^(mn)`, so a population that reaches `N/d_eff = 42.7` is affordable
where full rank tops out at `0.167`. Classical coupling arguments should have room to work
there for the first time. They do not.

| claim | measurement |
|---|---|
| `orthogonal_hd` vs uncoupled, rank 1 and 4 | cosine ratio 0.99–1.01, IQRs overlap at every `N`, σ and rank, out to `N/d_eff = 42.7` |
| the treatment was applied | a 512-member coupled block is an **exactly** orthonormal basis of `ℝ⁵¹²` (max off-diagonal Gram `0.0e+00`); the i.i.d. block reaches `0.215`; contractions differ by 1.4 relative |
| why | measured `cos ≈ √(N/d_ambient)`; every curve slope ½ on log–log; full rank at `N = 2¹⁴` predicts 0.1021 against 0.1013 measured |

**The mechanism, which is the transferable part.** Coupling leaves `E[εεᵀ]` and the pairwise
cross-moments `E[ε_ij ε_i'j] = δ_ii'` unchanged, inside a design block as well as across it.
It therefore cannot alter the variance of a *linear* functional of the population; only
higher-order joint structure was ever in play. The `N/d_eff` argument establishes that low
rank creates *room* for sample design, which is not the same claim as a mechanism existing.
Conflating those two is the error, and it is the reusable content of the section.

**Two supporting results that came out of the same sweep:**

- **Coupling's cost is two numbers**, and they differ by 180×: **+4.2%** per generation at
  rank 1, **+770%** at full rank, because the design dimension is `m` versus `m·n`. The
  full-rank figure is dominated by a reference FWHT butterfly being memory-bound, which is
  the first concrete argument for a real JAX FWHT kernel as separate work.
- **Mirroring is not a free win, and its sign flips with σ.** At rank 1 it is 1.67× *better*
  than i.i.d. at `σ = 1e-2` and ~1.4× *worse* at `σ = 1e-3`. It cancels the even part of `f`,
  which matters only once σ is large enough to sample curvature; below that, mean-centering
  already removes the constant and mirroring merely spends the population on half as many
  distinct directions. "Use antithetic sampling" is doing unexamined work in the ES
  literature.

**Scope.** E1 measured estimator quality, not task performance, on one transformer block. The
smoothing caveat in `docs/00-context.md` cuts both ways and is stated in the limitations
rather than hedged here. `docs/BACKLOG.md` records the decision to treat this as settled.

---

### C6 — The noise-structure axis, measured on a real fine-tuning task for the first time

**Status: all four sub-claims measured (the 2026-08-17 campaign, the
frozen-embedding arm, and the 2026-08-18 8-GPU demo;
`experiments/countdown/results/`).** The
experiment is E13. This is the claim the
two 2025 papers cannot make and cannot test against each other: Qiu et al. is full-rank
only, EGGROLL is welded to RWKV, and no implementation before this one can vary the noise
structure while holding the task, model, budget and seeds fixed.

The setup: Countdown (verifiable arithmetic reward) on Qwen2.5-0.5B, `N = 30` matched to
Qiu, bf16 compute under the accepted dtype policy, and the perturbation strategy swept
`Mirrored(SeedRegenerated())` vs `Mirrored(LowRank(r))`, `r in {1, 4, 16}`. One GRPO curve
(TRL, Qiu's published hyperparameters, cited, untuned by us) as a reference line at matched
GPU-hours, so a reader can place ES at all. GRPO is context, not the subject; C6 stands or
falls on the within-ES comparison.

Four sub-claims, each falsifiable and each a finding whichever way it lands:

- **C6a, cost at matched quality. MEASURED, and rank matches.** Twelve runs (four
  arms, three seeds) land in 0.149-0.162 held-out reward with within-arm seed
  spread covering the between-arm spread; rank 1 runs 2.41 s/update steady
  against full rank's 4.29, a 1.78x per-update advantage (steady state means
  generations 0 AND 1 excluded, both compilation-scale; the 2026-08-22 review
  caught gen 1 riding in the old cumulative totals, plot_e13.py has the
  corrected accounting). That is Qiu's result at 1.8x lower update cost
  with EGGROLL's trick, in a regime EGGROLL never tested, on a held-out metric.
  The results directory README carries the campaign detail and the decode caveats.
- **C6b, F5 predictivity. MEASURED, and F5 called the tie.** The naive reading of
  the rank axis says rank 1 samples in a 640x smaller space than full rank, so at
  `N = 30` it should dominate. F5's fitted curves say otherwise: the low-rank
  panel's intercept (the projection loss E1 priced at ~2.2x per member) almost
  exactly cancels the `d_eff` advantage, so at E13's operating point the
  prediction is near-parity, a spread of at most 2x in cosine (full 1.7e-4,
  r1 2.7e-4, r4 2.8e-4; slope 0.50 on every fitted curve, both shaping slices).
  The observed outcome is a statistical tie, which is what near-parity in
  estimator quality looks like through 500 sigma-scaled steps. E13 has no power
  to adjudicate the residual 2x (the insignificant observed gap even points the
  other way), so what ships is exactly this: F5 predicted quality-flatness ex
  ante where the dimension-counting intuition predicted dominance, and the task
  agreed. Extrapolation is one to three decades below E1's smallest measured
  `N/d_eff`, on curves whose slope never wavers from one half; E1 has no rank-16
  curve, so lr16 rides on the r1/r4 trend. `experiments/countdown/analysis_c6b.py`
  reproduces every number from committed artifacts.
- **C6c, embeddings under perturbation. MEASURED, and freezing costs nothing here.**
  Qwen's tied embedding is ~27% of the 0.5B parameters, and EGGROLL's reference
  raises NotImplementedError there; the `embed` seam perturbs it without forming
  the table, which is what makes the ablation runnable at all. Three seeds of
  rank 1 with the embedding frozen behind the model closure land at 0.154
  [0.149-0.157] held-out reward against the live arm's 0.155 [0.152-0.158]: on
  Countdown at this scale, perturbing and updating the tied table buys nothing
  detectable, so the capability's value here is the choice it enables (27% of
  the parameters can sit out for free), not a quality win. A task whose reward
  depends on token-level knowledge rather than arithmetic composition is where
  the two arms could still separate; that is a scope note, not a finding.
  `results/e13-a100-2026-08-17/es-lr1-frozen-embed-*` alongside the campaign.
- **C6d, reproducibility. MEASURED, and the claim sharpened into three.** At 0.5B
  on 8 real A100s: (1) same program, same seed is *bitwise* deterministic across
  processes (two independent 20-generation D=8 runs agree exactly in all 494M
  final parameters and every logged reward; no PPO/GRPO trainer offers this);
  (2) the update path is device-count invariant at tolerance (fixed fitness,
  D=1 vs D=8, 6.3e-06 norm relative error on the full tree); (3) end-to-end
  trajectory equality across *different compiled programs* is not a bf16
  property, and the demo measured why: greedy argmax flips near-ties under
  different XLA fusion/autotuning choices, so the D=1 and D=8 trajectories part
  at generation 0 even though a D=1 and a D=8 program were also observed
  producing bit-identical tokens for all 32 members. The divergence tracks the
  program, not the device count; an XLA upgrade would do the same. Stating this
  boundary is part of the claim.
  `results/c6d-a100x8-2026-08-18`, with the decomposition script beside it.

What is deliberately not claimed: beating GRPO on final reward. That is Qiu's result; at
pilot scale it may not reproduce, and no sub-claim above depends on it.


## Experiment matrix

Tiers are defined in `docs/06-benchmark-runbook.md`. Short version: **T0** CPU (free),
**T1** Kaggle TPU v5e-8 (free), **T2** the local RTX 3080 (free), **T3** TRC TPU, larger
slices (free), **T4** GCP paid GPU, **T5** neocloud spot GPU (cheap reruns).

| ID | Experiment | Claim | Tier | Est. hrs | Cost |
|---|---|---|---|---|---|
| **E0** | Correctness, device-invariance, comm accounting | C2 | T0 | **mostly done** | $0 |
| **E1** | Estimator quality: `N` × rank × scheme × shaping × σ | C5 | T2 | **done, 13.1** | $0 |
| **E2** | Strong scaling, TPU, `D ∈ {1,2,4,8}` | C3 | T1 | **done** (`results-tpu-v5e8`, one session) | $0 |
| **E3** | Weak scaling, TPU, `D ∈ {1,2,4,8}` | C3 | T1 | **done** (`results-tpu-v5e8`) | $0 |
| **E4** | Contraction crossover, TPU, `(N, d)` grid at `D=8` | **C1** | T1→T3 | **done at sweep resolution** (`figures-tpu-v5e8` M3; the denser grid below stays open for TRC) | $0 |
| **E5** | Scaling past 8 devices: `D ∈ {16,32,64}` | C1, C3 | T3 | ~15 | ~$10 |
| **E6** | Strong/weak scaling, GPU, `D ∈ {1,2,4,8}` | C1, C3 | T4 | **done** (docs/03 M1/M2, 8×A100) | spent |
| **E7** | Contraction crossover, GPU | **C1** | T4 | **done** (docs/03 M3, 8×A100) | spent |
| **E8** | Low-rank vs dense cost surface, TPU **and** GPU | **C4** | T1 + T4 | **done both** (`results-cost`, `results-cost-tpu-v5e8`; F4 drawn) | ~$3.50 |
| **E9** | Baselines: naive ES, EGGROLL ref impl, evosax | C2 | T4 + T1 | **done both** (docs/03 M4; `results-m4-tpu-v5e8`) | spent |
| **E10** | Shaping-barrier cost (global rank sort) | C1 | T1 | **done** (`results-barrier-tpu-v5e8`: gather 2.4-6 us, sort 12.1 ms at N=2^18, D-independent) | $0 |
| **E11** | Ablations: `r`, σ, dtype, accumulation precision | all | T1 | **mostly assembled** (`tb3.py` from E1/E8/E13 + tests); one session open (T4: precision ratio + lr1 re-measure) | $0 |
| **E12** | End-to-end task validation, ≥3 seeds | C2 | T3 | ~15 | ~$8 |
| **E13** | Countdown, Qwen2.5-0.5B: rank sweep + GRPO reference | **C6** | T2→T5 | ~30 | ~$30-80 |
| **E14** | Stability under hyperparameter perturbation: lr x{1/8,8}, sigma x{1/4,4}, kl_beta + clip ablations, 3 seeds | **C7** | T5 | **done** (`countdown/results/e14-a100-2026-08-20`; ES 0 collapses, GRPO 3; both ablations trained fine, reported) | ~$32 |
| **E16** | Correction portability: new prompt batch gate at 0.5B, then 1.5B with raw + calibrated predictions frozen | C6 | T5 | **done** (`results-e16-gate` PASS; `results-e16-stage2`: law transfers unrefitted, calibration within 25%) | ~$21 |
| **E17** | Contraction crossover on the real model, TPU v5e, D 1-8, A/B x seed/rank-1 | C1, C3 | T1 | **done** (`results-e17`: both signs transfer, widening in D; Table 5) | $0 |
| **E15** | Estimator accuracy on the real model: production update vs true NLL gradient, 5 strategies x N in {30,240}, 5 seeds | C6 | T5 | **done** (`countdown/results-e15`; full rank 0.91-0.94x of fit, low-rank uniformly ~0.5x, unpaired 1.36x vs mirrored against 1.34x predicted; Table 4) | ~$19 |

Roughly **150 free accelerator-hours** and **one paid 6-hour GPU session**.

### Notes on individual experiments

**E4 is the paper.** Grid it properly: `N ∈ {2⁶ … 2¹⁸}` × `d` spanning at least three model
sizes × `D ∈ {1,2,4,8}`, both strategies, ≥5 timed repeats. This is the phase diagram; it
deserves the most hours and the most care.

**E5 is what TRC is for.** A scaling curve to 8 devices is fine. A curve to 64 is a
different-caliber figure, and it's free. Requires TRC quota for a `v5e-32` or `v5e-64`
slice — see the application-timing warning in the runbook.

**E5 deepened (designed 2026-08-21, deferred: needs multi-host TPU, the paper does not
wait for it).** Not just longer scaling curves: take the contraction crossover across
the host boundary, because it tests C1 directly, closes the paper's first limitation,
and completes the conclusion's closing sentence. Design, fixed before any run:

- `D ∈ {8, 16, 32, 64}` on one v5e slice, strategies A and B interleaved, existing
  warm-up/repeat protocol, strong and weak scaling.
- Two perturbations bracketing the phenomenon: seed-regenerated full rank (A's
  replicated contraction is constant in D while B's shrinks 1/D, so B should keep
  winning) and rank 1 (contraction nearly free, so B's model-sized all-reduce buys
  almost nothing and the sign should flip to A).
- Three `(N, d)` cells pre-selected from M3's D=8 phase diagram: one A-favored
  (low-rank at d=2048), one near-crossover (low-rank d=512), one B-favored
  (seed-regenerated).
- One instrument, two outputs: isolated timings of the fitness gather, the
  contraction, and the model all-reduce (barrier.py's method extended), which are
  also the calibration data for an alpha-beta latency/bandwidth model. Calibrate at
  the smaller D, commit predictions for the D=32/64 cells before running them, then
  reveal. That turns the phase diagram into a decision model.
- Hypothesis committed in ICI-aware form: a v5e-16/64 slice is one 2D-torus fabric,
  so hosts are an orchestration boundary, not a bandwidth cliff; what grows with D
  is torus diameter and per-chip bisection. The pre-registered question is whether
  B's advantage survives collectives spanning hosts on one fabric, with a gentle
  trend an expected outcome, not a failed prediction. The sharp cliff is the slice
  (DCN) or GPU node boundary, out of scope for this grant.
- Either outcome publishes: a separation gives the practical rule (partial
  contraction for expensive perturbations, scalar gathering for cheap structured
  ones once collectives span hosts); B-winning-everywhere sharpens the folk-claim
  result.

Free rehearsal before any granted hour: the whole driver multi-process on CPU
(`jax.distributed.initialize`, several local processes, forced host devices), gated
on the device-count invariance test.

**GPU variant (added 2026-08-22, possible follow-up, needs no grant).** The
host-boundary half of the question does not need TPU at all, and GPUs are the
sharper instrument for it: NVLink inside a node (600 GB/s class) against
InfiniBand or Ethernet between nodes (roughly 12-50 GB/s) is a one-to-two
order-of-magnitude cliff exactly at the boundary, where a v5e slice is one ICI
torus end to end. Two nodes of 8xA100 (D=16, one boundary) suffice for the
core question: does B's advantage survive the model-sized all-reduce paying
inter-node prices while A's scalar gather stays negligible. Same design as
above (interleaved A/B, both bracketing perturbations, pre-selected cells,
component timings doubling as alpha-beta calibration, predictions committed
before the held-out cells), with two GPU-specific rules: measure and report
the actual inter-node link first (rented fabrics vary from IB to plain
Ethernet, and the decision rule must be stated in terms of measured link
parameters to be portable), and the same CPU multi-process rehearsal covers
this backend too (NCCL under jax.distributed, mesh code unchanged). Cost
ballpark $50-150 for a disciplined calibrate-predict-measure session, so this
half can run before any TRC grant; the TPU half then adds depth in D on one
fabric, and the calibrated model is what unifies the two.

**E8 needs matched shapes across platforms**, not matched memory. v5e has 16 GB/chip
against A100's 80 GB, so per-device population must be matched deliberately rather than
"whatever fits." State the matching rule in the paper.

**E11 is assembly, not a campaign.** Walked claim by claim (2026-08-19), every axis but
one is already measured somewhere committed: r by E1/E8/C6a, σ by E1, dtype cost by E8
(and it is ~1.0x at default precision on both platforms, so dtype is a memory choice),
dtype correctness by the bf16 policy doc and `tell`'s guard. `tb3.py` builds TB3 from
those directories. The one new number is what `highest` costs over `default` on identical
cells (cost-precision-tpu.yaml, session T4). Deliberately NOT run: task-level σ and dtype
sweeps, ~30 GPU-hours of pre-emptive reviewer defence; if a reviewer asks, each is a
scoped follow-up, and this line is where that decision is recorded.

**E9's honest framing**: being *within* EGGROLL's own throughput while offering a general
API is a good result. Report it that way. Faster is a bonus, not the claim.

**E12 no longer carries C5.** It was going to be the check that estimator MSE is not task
performance (the smoothing caveat in `docs/00-context.md`), on the rule that C5 had to survive
both E1 and E12 to go in as a positive claim. C5 did not survive E1, so E12 shrinks to its C2
half: both published algorithms running end to end from one API. That halves its budget.

The rule was right and is worth keeping for whatever claim comes next — a positive result that
only exists in estimator space does not go in the paper as a positive result.

**E13 is that next claim, and the rule binds it.** C6b is exactly a claim crossing from
estimator space to task space, so it ships only if the F5 ordering survives contact with
Countdown. Structure: the port of Qwen2.5-0.5B to the `dense`/`embed` seams is local, free,
and the long pole (validated against reference logits before any rented hour); the rank
arms are the cheap ones, so the expensive full-rank run is bought once per seed and the
sweep rides on low-rank pricing. Single A100-80GB throughout, no multi-GPU booking
anywhere in E13. Baseline discipline: GRPO hyperparameters are Qiu's published ones,
cited, untuned by us in either direction. Runtime estimates are FLOP arithmetic on
decode until the pilot's first hour turns them into measurements, and the pilot is
designed to be cheap to abort.

---

## Figures

| # | Figure | From | Role |
|---|---|---|---|
| F1 | Strong + weak scaling, TPU and GPU panels, ideal line dashed | E2, E3, E6 | Opening figure. **Exists**: `experiments/phase2/figures/f1-scaling.png` (`plot_paper.py`) |
| F2 | **Contraction crossover phase diagram** in `(N, d)` at `D=8`, one panel per platform | E4, E7 | **The money figure**. **Exists**: `experiments/phase2/figures/f2-crossover.png`. The sign flips inside the low-rank panels and moves with the platform |
| F3 | Scaling to 64 devices, TPU | E5 | Shows the design actually distributes |
| F4 | Low-rank vs dense cost surface, TPU vs GPU | E8 | C4. **Exists**: `experiments/phase2/figures/f4-cost-*.png`. Measured answer: no inversion, low-rank pays MORE on TPU |
| F5 | Estimator quality vs `N/d_eff`, full-rank / rank-1 / rank-4 panels | E1 | C5. **Exists**: `experiments/phase0/figures/` |
| F6 | End-to-end task curves, seed-variance bands | E12 | C2: both algorithms run end to end. *Was also C5 validation; dropped with C5.* |
| F7 | E13 held-out curves: four ES ranks + frozen embedding + GRPO, seed bands, base-model floor | E13 | **Exists**: `experiments/countdown/figures/`. C6 in one panel |
| F8 | Robustness curves: final held-out reward vs hyperparameter multiplier, per algorithm, collapse markers | E14 | C7. **Exists**: `experiments/countdown/figures/f8-robustness.png` (`plot_e14.py`) |
| TB1 | Baseline throughput table, matched shapes | E9 | C2 |
| TB2 | Communication accounting: analytic vs measured | E0, E4 | Rigor; catches bugs |
| TB3 | Ablation table | E11 | Reviewer defence |

Tables are `TBn`, not `Tn`, so they stay clear of the compute tiers `T0`–`T5` in
`docs/06-benchmark-runbook.md`. Both were `T1`/`T2`/`T3` and both appear in this file.

F2 is the figure to design first and work backwards from.

---

## Paper structure

1. **Introduction** — two 2025 papers, opposite bets, no common library, and an unexamined
   folk claim about ES communication.
2. **Background** — ES gradient estimator; the two algorithms; why `ravel_pytree`
   forecloses both.
3. **Design** — the perturbation-strategy abstraction; sample / apply / contract; the seed
   contract; sharded state.
4. **The contraction question** — Strategies A and B, analytic cost model, predicted
   crossover. *Written before the measurements, so the model is a prediction and not a
   post-hoc fit.*
5. **Experimental setup** — platforms, shape-matching rule, measurement protocol.
6. **Results** — F1–F4, TB1–TB3.
7. **Coupled sampling** (conditional) — F5, F6, or the negative result.
8. **No limitations section.** Decided 2026-08-23: a caveat goes with the claim it
   qualifies, in the same paragraph, or it goes nowhere. A separate section invites
   generic entries nobody expected otherwise ("not every configuration was measured")
   and lets a real caveat sit four pages from the number it bounds. The list below is
   the audit trail of where each one landed, not a section to write.
9. **Related work** — ES at scale, ZO optimization for LLMs (P-GAP, LOREN, GRZO), coupling
   and QMC for ES, sharding in JAX.

Writing §4's cost model **before** running E4 is a deliberate methodological choice. A
predicted crossover that the measurement confirms is a much stronger result than a curve
fitted afterwards, and it's cheap to do — write it during Phase 1.

---

## Caveats, and the claim each one travels with

- Single-node for the GPU results, if that's what the budget buys.
- **The cost model closes the dense side of the crossover and not the low-rank side.**
  Bytes reach time through the measured collectives (`docs/11-cost-model.md`, `tb7`):
  where B wins its all-reduce is 0.7-3.6% of the advantage, where A wins it is 57-155%
  of B's deficit. Eight of the eleven fall UNDER 100%, which is B costing more than its
  local contraction plus the isolated all-reduce allows. State the open term; do not write
  "the crossover is an observation, not a cost model", which stopped being true once
  the ladder ran.
- v5e-8 has 16 GB/chip, so absolute population sizes don't match EGGROLL's H100 figures.
  The scaling *behaviour* transfers; the absolute numbers don't.
- Kaggle T4s are Turing-class and appear only in correctness runs, never in a throughput
  claim.
- E12's tasks are small relative to a 14B RWKV. Say which conclusions are extrapolation.
- **C5's negative is about the estimator, not the optimizer.** Parameter-space noise acts as
  a Gaussian smoothing of a jagged reward landscape, so it is doing optimization work rather
  than only adding error, and a better-conditioned estimate can be a worse smoother. The
  classical QMC-for-ES wins are on multimodal control problems; a single transformer block is
  not one. State this as the boundary of the claim. Do not soften the negative with it — the
  measurement is clean within its scope, and the scope is the estimator.
- **C5 is one objective, one block, one hardware generation.** `N/d_eff` transfers; the loss
  landscape does not.
- **The scrambled-Sobol arm is reported as unresolved, not as a result.** It was the only
  scheme that separated from its baseline, and it separated the *wrong* way (cosine ratio
  0.892 at `N = 2¹⁸`, rank 1, IQRs disjoint, degrading monotonically in `N`). Lost design
  diversity and marginal-moment error are both ruled out with numbers; the cause is not
  established and may be the digital-shift construction rather than the method. Say that, and
  do not write "QMC hurts ES". `docs/BACKLOG.md` B1.
- **`σ = 0.1` and `shaping = none` are dead arms on this block**, measured, not assumed: they
  give `cos ~ 1e-3` and occasionally negative. Reported so a successor sweep does not spend a
  third of its grid on them.

---

## Venue and timing

**Framing decided 2026-08-21: the paper is the ES-scaling investigation; the library is the instrument**, cited and released, one contribution bullet. The project's deliverable is still the library (CLAUDE.md); the paper's contribution is what it measured, which is both the stronger review posture and the better story.

**Title, current: "Update-contraction placement in sharded evolution strategies on GPUs and TPUs"** (239a45a, 2026-08-22). It names the measurement. The previous title said "the contraction crossover", and only the low-rank family crosses: the dense and seed-regenerated families never change sign anywhere in the grid, so "crossover" promised something two thirds of the results do not show. "Crossover" stays in the body, where it names the low-rank sign flip and is accurate.

Superseded on the way here: "Sharded evolution strategies on GPUs and TPUs" (2026-08-21, plain and descriptive but says nothing about what was found), and before that the folk-claims phrasing (too informal), a communication-only title (undersells the accuracy results), and any title leading with shardes.

**Decided 2026-08-20: MLSys-style full paper** ("it can always be compressed later").
The fallbacks stay what they were: a NeurIPS/ICML workshop on efficient training or
systems, or arXiv preprint plus workshop. The artifact — a working library with a
two-minute test suite — is a strong component for artifact-evaluation tracks.

Deadline (looked up 2026-08-20, not recalled): MLSys 2027's is not yet announced; the
2026 cycle opened Sep 15 2025 and closed **Oct 30 2025**, so the same cadence puts 2027
at roughly **late October 2026**. Check mlsys.org when the CFP lands; do not build the
schedule on the extrapolation alone.

Rough arc, from Phase 0 start:

| Weeks | Work |
|---|---|
| 1–2 | E1 on the local RTX 3080; Gate G0 |
| 3–12 | Phase 1 library; E0 continuously on CPU; write §4's cost model |
| ~10 | **Apply to TRC** — timed so the grant window opens when the code is ready |
| 13–16 | E2, E3, E4, E8(TPU), E10, E11 on Kaggle + TRC |
| 17 | E5 on TRC; one paid GPU session for E6, E7, E8(GPU), E9 |
| 18–21 | E12 |
| 22–26 | Writing, artifact packaging, reruns |

TRC's grant is temporary and historically ~30 days. Applying too early wastes the window —
this is the opposite of the GPU-quota advice, and it's the most common way to lose the free
compute. Details in the runbook.
