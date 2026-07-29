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

### C2 — Both LLM-scale ES algorithms are expressible under one abstraction

Qiu et al. (full-rank, seed-regenerated, `N=30`) and EGGROLL (rank-`r` factored, never
materialized, `N` up to 2¹⁸) make opposite bets on the same tradeoff and have no common
library. `evosax` forecloses both by flattening solutions via `ravel_pytree`. We show a
perturbation-strategy abstraction under which the two are a two-line config diff, with no
global flattening and no loss of throughput versus each paper's own implementation.

Artifact claim. Established by implementation + baselines, not by a curve.

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

### C5 — Coupled sampling under low-rank perturbation (conditional on Gate G0)

Only if Phase 0 says yes. See `docs/04-phase3-coupling.md`. If G0 says no, this becomes a
short negative-result section, which is still worth including — the `N/d_eff ≳ 1` regime
argument is novel enough that a clean null is publishable inside a larger paper.

---

## Experiment matrix

Tiers are defined in `docs/06-benchmark-runbook.md`. Short version: **T0** CPU (free),
**T1** Kaggle TPU v5e-8 (free), **T2** Kaggle GPU P100/2×T4 (free), **T3** TRC TPU, larger
slices (free), **T4** GCP paid GPU, **T5** neocloud spot GPU (cheap reruns).

| ID | Experiment | Claim | Tier | Est. hrs | Cost |
|---|---|---|---|---|---|
| **E0** | Correctness, device-invariance, comm accounting | C2 | T0 | ∞ | $0 |
| **E1** | Estimator quality: `N` × rank × scheme × shaping × σ | C5 | T2 | ~20 | $0 |
| **E2** | Strong scaling, TPU, `D ∈ {1,2,4,8}` | C3 | T1 | ~12 | $0 |
| **E3** | Weak scaling, TPU, `D ∈ {1,2,4,8}` | C3 | T1 | ~8 | $0 |
| **E4** | Contraction crossover, TPU, `(N, d)` grid at `D=8` | **C1** | T1→T3 | ~20 | $0 |
| **E5** | Scaling past 8 devices: `D ∈ {16,32,64}` | C1, C3 | T3 | ~15 | ~$10 |
| **E6** | Strong/weak scaling, GPU, `D ∈ {1,2,4,8}` | C1, C3 | T4 | ~3 | in session |
| **E7** | Contraction crossover, GPU | **C1** | T4 | ~2 | in session |
| **E8** | Low-rank vs dense cost surface, TPU **and** GPU | **C4** | T1 + T4 | ~6 + ~2 | ~$0 + session |
| **E9** | Baselines: naive ES, EGGROLL ref impl, evosax | C2 | T4 + T1 | ~4 | in session |
| **E10** | Shaping-barrier cost (global rank sort) | C1 | T1 | ~4 | $0 |
| **E11** | Ablations: `r`, σ, dtype, accumulation precision | all | T1 | ~15 | $0 |
| **E12** | End-to-end task validation, ≥3 seeds | C2, C5 | T3 | ~30 | ~$15 |

Roughly **150 free accelerator-hours** and **one paid 6-hour GPU session**.

### Notes on individual experiments

**E4 is the paper.** Grid it properly: `N ∈ {2⁶ … 2¹⁸}` × `d` spanning at least three model
sizes × `D ∈ {1,2,4,8}`, both strategies, ≥5 timed repeats. This is the phase diagram; it
deserves the most hours and the most care.

**E5 is what TRC is for.** A scaling curve to 8 devices is fine. A curve to 64 is a
different-caliber figure, and it's free. Requires TRC quota for a `v5e-32` or `v5e-64`
slice — see the application-timing warning in the runbook.

**E8 needs matched shapes across platforms**, not matched memory. v5e has 16 GB/chip
against A100's 80 GB, so per-device population must be matched deliberately rather than
"whatever fits." State the matching rule in the paper.

**E9's honest framing**: being *within* EGGROLL's own throughput while offering a general
API is a good result. Report it that way. Faster is a bonus, not the claim.

**E12 is the one that can't be shortcut.** Estimator MSE is not task performance (see the
smoothing caveat in `docs/00-context.md`). If C5 survives E1, it has to survive E12 too, on
≥3 seeds with variance reported, or it doesn't go in as a positive claim.

---

## Figures

| # | Figure | From | Role |
|---|---|---|---|
| F1 | Strong + weak scaling, TPU and GPU panels, ideal line dashed | E2, E3, E6 | Opening figure |
| F2 | **Contraction crossover phase diagram** in `(N, d)` at `D=8`, one panel per platform | E4, E7 | **The money figure** |
| F3 | Scaling to 64 devices, TPU | E5 | Shows the design actually distributes |
| F4 | Low-rank vs dense cost surface, TPU vs GPU | E8 | C4; the cross-platform inversion |
| F5 | Estimator quality vs `N/d_eff`, rank-1 and full-rank panels | E1 | C5, conditional |
| F6 | End-to-end task curves, seed-variance bands | E12 | C5 validation |
| T1 | Baseline throughput table, matched shapes | E9 | C2 |
| T2 | Communication accounting: analytic vs measured | E0, E4 | Rigor; catches bugs |
| T3 | Ablation table | E11 | Reviewer defence |

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
6. **Results** — F1–F4, T1–T3.
7. **Coupled sampling** (conditional) — F5, F6, or the negative result.
8. **Limitations** — write this honestly and early; see below.
9. **Related work** — ES at scale, ZO optimization for LLMs (P-GAP, LOREN, GRZO), coupling
   and QMC for ES, sharding in JAX.

Writing §4's cost model **before** running E4 is a deliberate methodological choice. A
predicted crossover that the measurement confirms is a much stronger result than a curve
fitted afterwards, and it's cheap to do — write it during Phase 1.

---

## Limitations to state, not bury

- Single-node for the GPU results, if that's what the budget buys.
- v5e-8 has 16 GB/chip, so absolute population sizes don't match EGGROLL's H100 figures.
  The scaling *behaviour* transfers; the absolute numbers don't.
- Kaggle T4s are Turing-class and appear only in correctness runs, never in a throughput
  claim.
- E12's tasks are small relative to a 14B RWKV. Say which conclusions are extrapolation.
- If G0 was ambiguous, say so plainly rather than picking the reading that helps.

---

## Venue and timing

Realistic targets, in order of preference: an **MLSys**-style systems venue (the C1
crossover is the right shape), a **NeurIPS/ICML workshop** on efficient training or
systems, or **arXiv preprint plus workshop**. The artifact — a working library with a
two-minute test suite — is a strong component for artifact-evaluation tracks.

**Check current deadlines before committing to any of these.** Do not build a schedule
around a date recalled rather than looked up.

Rough arc, from Phase 0 start:

| Weeks | Work |
|---|---|
| 1–2 | E1 on Kaggle GPU; Gate G0 |
| 3–12 | Phase 1 library; E0 continuously on CPU; write §4's cost model |
| ~10 | **Apply to TRC** — timed so the grant window opens when the code is ready |
| 13–16 | E2, E3, E4, E8(TPU), E10, E11 on Kaggle + TRC |
| 17 | E5 on TRC; one paid GPU session for E6, E7, E8(GPU), E9 |
| 18–21 | E12 |
| 22–26 | Writing, artifact packaging, reruns |

TRC's grant is temporary and historically ~30 days. Applying too early wastes the window —
this is the opposite of the GPU-quota advice, and it's the most common way to lose the free
compute. Details in the runbook.
