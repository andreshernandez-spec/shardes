# Phase 2 — Scaling benchmarks

**Compute**: 8 GPUs, 4–6 hours. See `docs/compute.md` for how to get them and what it costs.
**Duration**: 2–3 weeks (mostly preparation; the run itself is one afternoon).
**Gate**: G2 — a scaling curve worth putting at the top of the README.

---

## Goal

Produce the numbers the library exists to produce. Phase 0 measured statistical efficiency
on one GPU; this measures **systems efficiency**, which is unmeasurable without the library
and is the actual headline.

---

## What gets measured

### M1 — Strong scaling
Fixed total population `N`, devices `D ∈ {1, 2, 4, 8}`. Report wall-clock per generation
and parallel efficiency `T₁ / (D · T_D)`.

The honest expectation: ES should scale close to linearly, because the only mandatory
cross-device traffic is `N` fitness scalars. If it doesn't, the interesting result is
*why* — and the two candidates are the shaping barrier (C1.6) and the contraction strategy
(C1.3). Both are instrumented, so you'll be able to say which.

### M2 — Weak scaling
Fixed population **per device**, scale `D`. Report throughput and per-device memory. This
is the plot that shows the design actually distributes.

### M3 — The contraction crossover ⭐
Strategy A (scalar all-reduce + replicated regeneration) vs Strategy B (model-size
all-reduce of the partial update), swept over `N` and `D` at two model sizes.

This is the most publishable single result in the phase, because the folk claim "ES only
all-reduces scalars" is conditional and nobody has drawn the boundary. Output: a phase
diagram in `(N, d)` with the crossover contour, at `D = 8`.

### M4 — Against the references
- **Naive ES** (materialize every perturbation) — the baseline both papers beat.
- **EGGROLL's own JAX implementation**, same GPU, same shapes, same `N`. Being *within* its
  throughput while offering a general API is a perfectly good result and should be reported
  as such. Being faster is a bonus, not the claim.
- Optionally `evosax` at whatever `N` it manages before the flattening bites.

Report tokens/s or env-steps/s at a stated parameter count, matching how both papers report.

### M5 — Communication, instrumented
Bytes moved per generation, measured, next to the analytic prediction from Phase 1. Include
the shaping barrier's contribution separately.

### M6 — Memory
Peak per-device memory vs `D`, showing perturbation storage and state falling as `1/D`.
Note where `A`/`B` storage (`N·r·(m+n)`) becomes the binding constraint — that's the
threshold that motivates seed regeneration inside the low-rank path.

---

## How to test it

Benchmarks lie more easily than tests do. Guards:

| Guard | What it prevents |
|---|---|
| Warm-up generations discarded (≥3) | Measuring JIT compilation |
| `block_until_ready()` on every timed result | Measuring async dispatch, not compute |
| ≥5 timed repeats, report median + IQR | Single-sample noise |
| Fixed seed; assert the optimizer trajectory is identical across `D` | Benchmarking a *different computation* at each `D` |
| Config committed before the run, cited by SHA in results | Post-hoc config tuning |
| Roofline sanity check on at least one kernel | Claiming a speedup that's above hardware limits |
| Same shapes for every method compared | The classic accidental apples-to-oranges |

The trajectory-identity check deserves emphasis: if the 8-device run takes a different
optimization path than the 1-device run, the scaling number is meaningless. `Phase 1`'s
`test_device_invariance` is what makes this assertable.

---

## How to showcase it

**The README's first figure** is M1 + M2: strong and weak scaling, `D` on the x-axis, ideal
line dashed. If this figure is good, it does most of the work of the whole project.

**The second figure** is M3, the crossover phase diagram. This is the one that reads as
"understands distributed systems" rather than "implemented a paper."

**A results table**: throughput at a stated parameter count, versus naive ES and versus
EGGROLL's implementation, with the hardware and JAX version in the caption.

**A written honest-limitations section**: single-node only (if it is), which parts are
extrapolation, what the shaping barrier costs, and where the contraction crossover puts
this library's sweet spot.

Deliverables live in `experiments/phase2/` with a README recording GPU model, driver, CUDA
version, JAX version, commit SHA, and total spend.

---

## The 4–6 hour run: how not to waste it

Rented multi-GPU time is the one genuinely irreversible cost in this project. Treat the
session as an execution of an already-debugged plan, not as a debugging session.

**Two weeks before** — request GPU quota if going the GCP route (`docs/compute.md`; this is
the single biggest scheduling risk).

**The week before** — a full dress rehearsal on 1–2 GPUs:
- every benchmark configuration runs to completion with `N` reduced 100×,
- results are written to disk incrementally, one file per configuration,
- the driver is resumable: re-running skips completed configs,
- there is a hard wall-clock cap per configuration, and exceeding it logs and moves on,
- the plotting script runs end-to-end on the rehearsal data and produces the final figures.

**The day before** — build the exact container image, push it, and boot a single-GPU
instance from it to confirm JAX sees the GPU and the driver starts. Image build failures on
a rented 8-GPU box are the classic way to burn $150 on `pip install`.

**During** — run the sweep in priority order: M1 and M2 first (they're the README figure),
then M3, then M4. If time runs out, you want to lose the least important measurement, not a
random one.

**Ordering matters more than duration.** A 6-hour budget with the right ordering beats a
12-hour budget without it.

---

## Exit criteria — Gate G2

1. Strong and weak scaling curves across `D ∈ {1,2,4,8}`, with parallel efficiency stated.
2. The contraction crossover measured, with the phase diagram.
3. At least one comparison against an external reference at matched shapes.
4. Every number reproducible from a committed config + a recorded environment.
5. A limitations paragraph that a skeptical reader would accept as fair.

If the scaling is worse than expected, **that is still a passing gate** provided the cause
is identified and measured. "ES scales at 71% efficiency to 8 GPUs, and here is the
breakdown of where the other 29% goes" is a better artifact than an unexplained 95%.
