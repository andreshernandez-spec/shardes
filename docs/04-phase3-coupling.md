# Phase 3 — Coupling at scale

**Conditional on Gate G0.** If Phase 0 said "no", skip this and reclaim the month — say so
in the README and move on. If it said "ambiguous", revisit after G2 with real tasks rather
than starting here.

**Compute**: 8 GPUs, 4–6 hours (a second session, or the tail of the Phase 2 booking).
**Duration**: 3–4 weeks.

---

## The claim being tested

EGGROLL inverts the `N/d` arithmetic that kills sample design in high-dimensional ES. Under
rank-1 perturbation you sample `a ∈ ℝᵐ` and `b ∈ ℝⁿ`, so per-layer sampling dimension is
`m + n`, not `mn`. At `N = 2¹⁸` and a 4096-wide layer, `N/d_eff ≈ 32` instead of `10⁻³`.

**Hypothesis**: coupling `{aₙ}` across `Sᵐ⁻¹` and `{bₙ}` across `Sⁿ⁻¹` tightens the
constant in EGGROLL's `O(1/r)` convergence to the full-rank update, because that rate
depends on how evenly `N` rank-1 outer products tile the matrix space.

Background and the derivation of the `N/d` scaling: `docs/00-context.md`.

---

## Capabilities delivered

### C3.1 — `Coupled` as a strategy wrapper, not a fork

```python
Coupled(LowRank(r=1), kind="orthogonal_hd")
Coupled(LowRank(r=1), kind="sobol_scrambled")
```

If Phase 1's abstraction was right, this is a wrapper and nothing in the sharded core
changes. If it requires touching the core, the Phase 1 API was wrong and that is itself a
finding worth recording.

### C3.2 — Coupling that survives sharding

The hard part, and the reason this is Phase 3 rather than Phase 0 extended.

Coupling is a **global property of the point set**. Sharding splits the point set across
devices. Orthogonalizing `{aₙ}` across all `N` members requires either communication or a
construction that is orthogonal *by construction* without a global operation.

- `orthogonal_hd` is the good case: `HD₁HD₂D₃…` applied to a per-device block is
  block-orthogonal with no communication, and blocks are mutually near-orthogonal by
  concentration. Cost is `O(d log d)` locally.
- `sobol_scrambled` needs **skip-ahead addressing** so device `k` generates points
  `[kN/D, (k+1)N/D)` independently. This is what makes QMC parallelize at all. There is no
  QMC in JAX — verified zero matches for `sobol|halton|scrambl|digital_net` in library
  code, `jax-qmc`, `jaxqmc`, `qmcjax`, `jax-sobol`, `sobol-jax` are all 404 on PyPI, and
  jax-ml/jax#8807 ("Implement quasi-random generators for JAX") was closed in Dec 2021 —
  so this is built from scratch.

  It is cheaper than it first looks, for two reasons worth writing down.

  Skip-ahead is only hard in the *sequential Gray-code* formulation, which is what most
  reference implementations use (including NVIDIA's CUDA sample) because it is fastest for
  emitting points `0…N` in order. The direct formulation, where point `i` is the XOR of the
  direction vectors selected by the set bits of `gray(i)`, is inherently random-access.
  Skip-ahead is not a feature added to it, it is the only thing it does, and it is the
  formulation that vectorizes anyway.

  Unbiasedness needs only a random digital shift, not full Owen nesting: XOR each point
  with one uniform random integer per dimension. That maps onto the seed contract exactly:
  **the shift derives from `base_key` and is shared across devices, the point index is the
  global member index.** Device-count invariance then falls out of the construction rather
  than being something to test for afterwards.

  The scarce artifact is the direction numbers, not the algorithm, and those are public:
  `scipy.stats.qmc.Sobol` carries Joe-Kuo to 21,201 dimensions. Dump the table once and
  embed it.

  Not worth binding to cuRAND (`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`, 20,000 dimensions,
  real `curandSetGeneratorOffset` skip-ahead) despite it being a good fit on paper: it is
  CUDA-only and the primary benchmark platform is TPU, and wrapping it means a JAX FFI
  custom call plus a compiled extension, which breaks the `pip install git+…@SHA` install
  path that Kaggle depends on. Check before using its *scrambled* variants directly in any
  case: the scramble constants appear to come from a fixed precomputed table, which would
  make them deterministic and therefore biased at fixed `N`. Unverified, and avoidable by
  passing your own constants, but the estimate feeds SGD so unbiasedness is not optional.

The device-count-invariance invariant (`CLAUDE.md`) still holds: coupled sampling must give
the same point set regardless of `D`.

### C3.3 — Validation on task performance, not MSE

**This is the whole methodological point of putting it here rather than in Phase 0.**

Lower estimator variance does not straightforwardly mean better ES. Parameter-space noise
acts as a Gaussian smoothing of a jagged reward landscape, so the noise is doing
optimization work, not just adding error. A better-conditioned estimate can be a worse
smoother, and coupling narrows exploration in a way that may hurt on multimodal objectives
— which is exactly where the classical QMC-for-ES results are *strongest*, so the sign is
genuinely uncertain.

Validation therefore means end-to-end runs at matched compute on:
- a multimodal control task (where the classical literature predicts coupling helps),
- a reasoning fine-tune (Countdown or GSM8K, matching both papers' setup),
- with ≥3 seeds each and the variance reported, since run-to-run stability is one of the
  things ES is claimed to be good at.

If the estimator improves and task performance doesn't, **report that**. It's a more
interesting result than a win, and it's the honest read of the smoothing argument.

---

## How to test it

| Test | Asserts |
|---|---|
| `test_coupled_unbiased` | `E[ĝ]` → `∇f` for every `kind`. Scrambling is what makes this true for Sobol; deterministic Sobol will fail it. |
| `test_coupled_device_invariant` | Same point set for `D ∈ {1,2,4,8}` simulated devices |
| `test_hd_block_orthogonality` | Per-device blocks orthogonal within tolerance; cross-block cosines within the `O(1/√d)` concentration band |
| `test_sobol_skip_ahead` | Device `k`'s points == the corresponding slice of the sequential sequence |
| `test_coupled_reduces_to_iid` | With coupling disabled, bitwise-identical to the uncoupled strategy |
| `test_wrapper_does_not_touch_core` | Static: `Coupled` introduces no new collective in the update path |

---

## How to showcase it

1. **The Phase 0 figure, re-run at scale** — same axes, now with real `N` on 8 GPUs, so the
   `N/d_eff ≳ 1` regime is reached with a realistic layer width rather than a scaled-down
   one.
2. **Task-performance curves** with seed variance bands, at matched compute. This is the
   plot that decides whether the idea is real.
3. **A cost table**: wall-clock overhead of coupling per generation. If `orthogonal_hd`
   costs 3% and buys nothing, that's the finding.

---

## Adjacent things this opens, deliberately out of scope

Recording these so they don't leak into the schedule:

- A proper **FWHT for JAX**. The reference butterfly written in Phase 0 is not a Mosaic GPU
  kernel. A real one has three independent consumers — LLM quantization rotations
  (QuaRot/QuIP#/SpinQuant), SRHT for randomized linear algebra, and orthogonal ES — and JAX
  ships only the dense `O(n²)` `jax.scipy.linalg.hadamard` constructor. That's its own
  project.
- A **QMC library for JAX**. Also entirely missing, also its own project, and a JAX
  maintainer has already blessed the idea existing outside JAX core.
- **Importance mixing** (classical NES sample reuse) — zero hits across the JAX ecosystem.
  Cheap to add, orthogonal to this phase.

`sobol_scrambled` ships in Phase 0 alongside `orthogonal_hd` rather than trailing it. The
direct formulation in C3.2 makes skip-ahead free, so there's no ordering dependency between
the two. What stays out is a separate project: direction numbers past 21k dimensions, Owen
nesting, higher-order digital nets, lattice rules.
