# Phase 3 — Coupling at scale — **DROPPED**

> **Gate G0 came back "no" on 2026-07-30, so this phase does not happen.** The condition
> below is met in the negative: skip it, reclaim the month, say so in the README.
>
> `orthogonal_hd` showed no estimator-quality separation at any rank, sigma or population,
> out to `N/d_eff = 42.7`, with the treatment verified maximal. The full answer and the
> reasoning are in `docs/01-phase0-estimator-harness.md`, "The answer: no".
>
> **The rest of this file is kept as written, not rewritten.** It is the record of what was
> predicted and why, which is the only thing that makes the negative result legible. Two
> parts of it did land and are live in the library: C3.1 (settled in Phase 0 — coupling is a
> noise source, not a wrapper) and the C3.3 cost table (measured in E1). C3.2, the sharded
> coupling work, is what got dropped, and it was the bulk of the three weeks.
>
> What survives as open questions is in `docs/BACKLOG.md`: B1 (why Sobol degrades), B2 (a
> real FWHT kernel), B3 (does coupling help an optimizer on a multimodal objective — which
> G0 did **not** answer).

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

### C3.1 — Coupling as a noise source, not a wrapper — **settled in Phase 0**

The check this capability set up was: *if the abstraction is right this is a wrapper and
nothing in the sharded core changes; if it requires touching the core, the API was wrong and
that is itself a finding worth recording.* Both halves came back, and they disagree, so here
is the finding.

**A wrapper is not constructible. The core is still untouched.**

```python
IIDGaussian(coupling=OrthogonalHD())          # not Coupled(IIDGaussian(), ...)
Mirrored(LowRank(r=1, coupling=OrthogonalHD()))
```

`Mirrored` gets to be a wrapper because antithetic sampling only touches **signs**: it
reuses the inner perturbation untouched and folds the sign into `contract`'s weights, so the
inner stays opaque. Coupling changes the **directions**, and a wrapper would have to reach
inside the inner perturbation and replace the noise it had just drawn. Three separate things
block that:

- The perturbation is opaque by protocol, and only the strategy knows which of its arrays
  are design axes. `LowRank`'s are the `(m, r)` and `(k, r)` factors, never the `(m, k)`
  product; `IIDGaussian`'s is the flattened leaf. A wrapper would need both layouts, and the
  next strategy's too.
- `SeedRegenerated` materializes nothing. There is no array to reach into, by design.
- HD coupling does not *transform* iid noise, it *replaces* it. Rows of `HD₁HD₂D₃` are built
  from Rademacher signs and consume no Gaussian, so drawing one first is wasted work rather
  than an input. QR of an iid block would be a genuine transform, and it is `O(d³)`, which is
  the reason HD exists.

So coupling is a constructor argument on the strategy: `Coupling` is a small protocol
(`shardes/coupling.py`) that replaces "draw `d` iid normals for member `i`" with "give member
`i` its share of a point set designed across members". `Gaussian` is the uncoupled default
and is bitwise what the strategies did before. What moved is one line inside each `sample`.

**This does not weaken the Phase 1 API, and it is worth being precise about why.** The
`sample`/`apply`/`contract` split is unchanged, `Perturbation` stays opaque, and no collective
was added. What the exercise showed is that the protocol was carved one level too high to
express sample design: the three methods describe *what a member's perturbation does*, and
coupling is a statement about *how members relate*. That is a different axis, and it gets its
own seam rather than a fourth method.

Two corrections to what this file used to claim, both from measurement:

- `HD₁HD₂D₃` is **exactly** orthogonal, not orthogonal within an `O(1/√d)` band. `H/√d` is
  orthogonal and symmetric and every Rademacher `D` is orthogonal, so the product is too.
  The `O(1/√d)` band belongs to cross-block pairs and to how close a row is to Haar. The old
  weak assertion would have passed on a broken chain.
- Coupling leaves `E[εεᵀ] = I` and the pairwise cross-moments `E[ε_ij ε_i'j] = δ_ii'`
  **unchanged**, inside a block as well as across. So it changes neither unbiasedness nor the
  variance of a *linear* functional of the population. Any leverage has to come from the
  higher-order joint structure the fitness nonlinearity sees. That sharpens Gate G0 rather
  than threatening it, but it means the effect cannot be argued into existence, and a null
  result is more likely a priori than the `N/d_eff` framing alone suggests.

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

Everything except device invariance landed in Phase 0, in `tests/test_coupling.py`, because
the coupling is a pure function of `(stream, member_id)` and needs no devices to test.

| Test | Asserts | Status |
|---|---|---|
| `test_unbiased_on_the_quadratic` | `E[ĝ]` → `∇f` for every coupled entry in the registry. Scrambling is what makes this true for Sobol; deterministic Sobol will fail it. | Phase 0, via `tests/test_estimator.py` over `STRATEGIES` |
| `test_hd_block_is_exactly_orthogonal` | A whole block's directions are orthonormal to float precision. **Not** "within a band" — see C3.1. | Phase 0 |
| `test_hd_blocks_are_independent_and_only_near_orthogonal` | Cross-block cosines inside the `O(1/√d)` band, and **not at zero**: zero would mean the same block was reused for every device. | Phase 0 |
| `test_a_members_draw_depends_only_on_its_global_id` | Block is `i // d`, position `i % d`. Device invariance follows from this without a device. | Phase 0 |
| `test_passing_the_default_coupling_changes_nothing` | Coupling off is bitwise the uncoupled strategy. | Phase 0 |
| `test_coupling_lands_on_the_lowrank_factors_not_the_product` | The design axis under low rank is `a ∈ ℝᵐ`, not `E ∈ ℝ^{mk}`. Coupling the product would still be unbiased and would sample in the wrong space. | Phase 0 |
| `test_sobol_skip_ahead` | Point `i` == element `i` of scipy's sequential sequence, for scattered `i`. | Phase 0 |
| `test_coupled_device_invariant` | Same point set for `D ∈ {1,2,4,8}` simulated devices | **Phase 3** |
| `test_no_new_collective_in_the_update_path` | Static: coupling adds no collective. Cheap now that it is a constructor argument rather than a wrapper. | **Phase 3** |

---

## How to showcase it

1. **The Phase 0 figure, re-run at scale** — same axes, now with real `N` on 8 GPUs, so the
   `N/d_eff ≳ 1` regime is reached with a realistic layer width rather than a scaled-down
   one.
2. **Task-performance curves** with seed variance bands, at matched compute. This is the
   plot that decides whether the idea is real.
3. **A cost table**: wall-clock overhead of coupling per generation. If `orthogonal_hd`
   costs 3% and buys nothing, that's the finding.

   **Measured in Phase 0, and the overhead is two numbers rather than one.** RTX 3080, one
   replicate, `m = n = 512`, `chunk = 256`, seconds per replicate:

   | strategy | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ | 2¹⁸ |
   |---|---|---|---|---|---|
   | `iid_gaussian` | 0.139 | 0.557 | 2.222 | 8.962 | 36.587 |
   | `seed_regenerated` | 0.199 | 0.763 | 3.045 | 12.470 | 49.170 |
   | `lowrank_r1` | 0.044 | 0.169 | 0.676 | 2.693 | 10.825 |
   | `mirrored_hd_lr1` | 0.045 | 0.178 | 0.712 | 2.804 | 11.283 |
   | `mirrored_sobol_lr1` | 0.044 | 0.170 | 0.695 | 2.754 | 10.791 |
   | `mirrored_hd_full` | 1.216 | 4.867 | 19.489 | 77.929 | **>400** |

   | | design dim | coupling overhead |
   |---|---|---|
   | rank 1, `orthogonal_hd` | 512 | **+4.2%** |
   | rank 1, `sobol_scrambled` | 512 | **−0.3%** (within noise) |
   | full rank, `orthogonal_hd` | 262,144 | **+770%** |

   The `3%` guess above is almost exactly right for the panel Gate G0 is about and wildly
   wrong for the other one. Coupling is free where it has leverage and ruinous where it does
   not, which is a happier coincidence than it had any right to be.

   `>400` is a probe timeout, so it is a lower bound. It is also why the full-rank population
   axis stops at 2¹⁴: 30 replicates of that cell need >3.3 h, and the 900 s cap would have
   recorded 2 replicates with an IQR. Reproduce with `experiments/phase0/`.

   **The cost is my reference FWHT, not the construction.** `transforms/fwht.py` is an
   18-stage butterfly that materializes a fresh array per stage, so it is memory-bound at
   ~9.6 GB of traffic per leaf per chunk against a forward-pass GEMM that is compute-dense and
   ~4× cheaper. Under full rank the design dimension is the whole `512×512` leaf, so every
   member pays two length-2¹⁸ transforms per leaf and the coupling costs more than the model.
   Under rank 1 it is `d = 512` and disappears into the noise.

   This is the first concrete number behind the "a real FWHT for JAX is its own project"
   claim below. It is a *reason* for that project, not a defect to fix mid-experiment:
   optimizing it now would change what E1 measures.

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
