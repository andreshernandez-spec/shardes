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

## Results, 2026-08-06

**The run.** 256 configurations, 0 failed, 0 over cap, 2h56m on 8x A100-SXM4-80GB (driver
595.71.05, CUDA 13.2, jax 0.11.0, community cloud, $27.90). `XLA_FLAGS` carried
`--xla_gpu_deterministic_ops=true --xla_gpu_enable_command_buffer=`; the second is not
optional on a multi-GPU node and `run.py` now refuses without it, see
`docs/06-benchmark-runbook.md`. Data in `experiments/phase2/results/`, figures in
`experiments/phase2/figures/`, guard output from `check.py`.

The figures were drawn by `plot.py` under **matplotlib 3.11.1**. Nothing records that
automatically: `env.json` and the per-result `env` block cover the numbers, not the
plotting, so a figure regenerated under a different matplotlib can differ from the committed
one with nothing flagging it. Worth fixing when the plotting path is next touched.

Preflight cost about a dollar and was worth it: `cudagraph.py` on 2x A100 found that every
`D>1` configuration dies in CUDA graph capture without the command-buffer flag. Undetected,
the 8-GPU session would have produced 64 single-device results next to 192 errors and exited
0.

**M1, strong scaling. There is essentially none.** Parallel efficiency `T1/(D*T8)` at `D=8`
lands between **0.112 and 0.142** across all 32 strong-scaling groups, and wall clock per
generation is flat or slightly worse as devices are added. It does not depend much on the
strategy or on the contraction: the best group is `seed_regenerated/B` at 0.142, the worst
`mirrored_lr1/A` and `lowrank_r1/A` at 0.112.

**M2, weak scaling.** Throughput at `D=8` is **0.99x to 1.55x** of `D=1` against an ideal of
8x, so the same picture from the other side: adding devices at fixed population per device
buys almost nothing.

**The cause is not yet identified, and that matters for the gate.** This section names two
candidates, the shaping barrier (C1.6) and the contraction strategy (C1.3), and says both
are instrumented so it will be possible to say which. Neither has been measured: **M5 was
not run.** Until it is, "scaling is flat" is an observation without a mechanism, which is
exactly what the gate below says is not good enough. The contraction is unlikely to be the
whole story, since A and B differ by at most 14% (M3) while the shortfall is a factor of
seven.

**M3, the contraction crossover.** `log10(t_B/t_A)` at `D=8` spans **-0.057 to +0.034**, so
the two strategies are within ~14% of each other everywhere in this grid. B is faster in 10
of 16 cells. **The four perturbation strategies disagree about the sign**: `mirrored_lr1`
favours A in every cell, `seed_regenerated` favours B in every cell, `iid_gaussian` and
`lowrank_r1` are mixed. The figure is faceted by strategy for that reason; collapsing them
produces a single confident answer that is not in the data.

No crossover contour is drawn. Two model sizes by three populations, with holes, cannot
carry one: any zero contour would be interpolation between four filled cells. The cell
values are the result at this resolution, and a denser `(N, d)` grid is what a contour needs.

**M6, memory.** `seed_regenerated` is flat at **15 MiB per device** across every device
count, against **12810 MiB** for `iid_gaussian/A` at `D=8` in weak mode, which is the
concrete case for seed regeneration and the clearest systems result in the sweep. Strategy A
storage does not fall with `D` in weak mode, as designed: every device regenerates the whole
population, so per-device storage tracks total `N`. One anomaly is unexplained: `iid_gaussian/A`
reads 1610 MiB at `D=1` and 810 MiB at `D=2` before rising, which is not monotonic and is
worth a look before the number is quoted.

**Device-count invariance.** 30 of 32 strong-scaling groups clean: 15 of 16 strategy A rows
bitwise identical across `D`, every B row at ~1e-07 against a 1e-5 tolerance. The exception
is `lowrank_r1` at `d=512, N=1024`, A and B, at 4.4e-05, and `check.py` classifies it as the
noise floor rather than a contraction bug from the recorded rank digests.

**The noise floor is wider than that one group.** `experiments/phase2/noisefloor.txt` is the
full table on one A100: **29 of 44 shapes cannot separate their closest two members by 16
ulp.** Everything at `d=512` from `N=128` up is under the margin, and at `d=2048` everything
from `N=64` up. Being under the margin means the ordering *can* be decided by rounding, not
that it *was*: only one group actually diverged. The right reading is that most of this
sweep's passing invariance results are fragile rather than robust, and would not survive a
different GPU, a newer XLA, or TF32.

**Draft limitations paragraph** (G2 criterion 5, for Andres to rewrite in his own words):

> Scaling was measured on 8x A100-SXM4-80GB at two model sizes, `d=512` and `d=2048`, with a
> 4-layer transformer block at batch 8 and sequence 32. At those shapes a generation costs
> 11 to 445 ms, so the measurement is plausibly dominated by per-device overhead rather than
> by the work being distributed, and the flat strong-scaling curve should be read as a
> statement about this operating point rather than about the design. The communication
> instrumentation (M5) that would separate the two was not run. Device-count invariance holds
> for 30 of 32 groups; the exception is a configuration whose population cannot be separated
> in float32, and 29 of 44 shapes in the sweep are close enough to that boundary that their
> agreement across device counts is not robust to a change of hardware or compiler. No
> external reference was measured, so no claim is made about performance relative to EGGROLL
> or evosax.

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

### Status after the 2026-08-06 run

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met** |
| 2 | crossover measured, phase diagram | **met**, without a contour: the grid is too coarse to carry one |
| 3 | one comparison against an external reference | **not met**, M4 was not run |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | drafted above, needs rewriting |

**The gate is not passed**, on 3 and on the clause above. Scaling came in at 0.11 to 0.14
efficiency, far worse than expected, and the escape hatch that makes that acceptable is
conditional on identifying the cause. The cause is not identified, because M5 was not run.
Two measurements stand between here and G2: M5, which says where the time goes, and M4,
which says whether any of this is competitive.
