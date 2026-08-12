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

> **Superseded as a description of current behaviour by "Results, 2026-08-11" below, and
> kept as the before half of a before and after.** Every number here was correctly measured
> on a program whose evaluation was replicated on every device
> (`docs/diagnosis-replicated-evaluation.md`). Two claims in this section are now known to be
> wrong rather than merely dated: the estimate that post-fix bandwidth would stay "under 1%
> of NVLink" (measured 1.43%), and the M1 figure, which was drawn from one of the four
> `(d_model, population)` blocks because of a key bug in `plot.py`.

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

**M2, weak scaling.** Throughput at `D=8` is **1.01x to 1.58x** of `D=1`, median 1.20x
across 32 series, against an ideal of 8x. The same picture from the other side: adding
devices at fixed population per device buys almost nothing.

**The cause is identified: the evaluation is not distributed at all.**
`experiments/phase2/profile.py --static` reports the FLOPs of the compiled per-device
program, and under SPMD that is what a device actually executes, so a computation that
distributes has them fall as `1/D`. They do not fall. In **all 16 configurations**, across
every strategy, both contractions and both model sizes, per-device eval FLOPs at `D=8` are
**identical** to `D=1`, ratio 1.000. Every device evaluates the whole population.

That accounts for the magnitude and not merely the direction. If per-device work does not
fall then `T_D = T_1`, and parallel efficiency `T_1/(D·T_D)` is exactly `1/D`. M1 measured
**0.112 to 0.142** at `D=8` against `1/8 = 0.125`. There is nothing else left to explain.

It also explains why the two contraction strategies barely differ (M3, within 14%) and why
communication is irrelevant (M5, 0.3% of NVLink): both describe what happens *after* an
evaluation that was already replicated `D` times.

**What this does not say.** Why `apply` is not sharding the population is a question about
the sharding logic, and `experiments/` is the wrong place to answer it. The measurement is
here; the diagnosis belongs with the library. Table in `experiments/phase2/profile.txt`.

`profile.py` also carries wall-clock columns, isolating `eval`, `tell`, the shaping barrier
and a dispatch floor. Those need real devices and were not run: 8-GPU stock was out on the
day, and the FLOPs result settles the question without them. Run them on the sweep hardware
if a breakdown of the remaining time is wanted.

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
population, so per-device storage tracks total `N`. `iid_gaussian/A` at `d=512` doubles
cleanly, 1610 to 3210 to 6410 to 12810 MiB at `N/device=128`.

An earlier revision of this section reported that series as non-monotonic (1610 at `D=1`,
810 at `D=2`) and called it unexplained. **It was a plotting bug, not a measurement.**
`plot.py` keyed weak-scaling series on `(strategy, how)` with neither the model size nor the
population per device in the key, so four experiments collapsed onto one line and whichever
sorted last won each device count; the "anomaly" was `N/device=128` and `N/device=32`
interleaved. Split correctly, 31 of 32 weak-mode memory series are monotonic in `D`. The one
that is not is `iid_gaussian/B` at `d=512, N/device=32`, which drops 12% from 486 to 426 MiB
between `D=1` and `D=2` and then doubles. That step is also where the program goes from
unsharded to sharded, so a different compiler choice is the obvious suspect, and it has not
been checked.

**M5, communication.** `experiments/phase2/comms.py` counts the payload of every collective
in the compiled HLO, and `experiments/phase2/comms.txt` is the table. Payload, not wire
bytes: a ring all-reduce moves roughly `2(D-1)/D` times the payload on the physical links,
so the wire figure is a constant factor away and depends on the algorithm XLA picks. The
payload is the part that is a property of the design.

**`docs/02` C1.3 is confirmed for B and undercounts A.** The table was re-measured on
2026-08-07 after `ShardedES.apply` began constraining its output; the numbers below are the
current ones, and the pre-fix table they replace is described in `comms.txt`'s header.

| | measured per generation | prediction | ratio |
|---|---|---|---|
| strategy A | 1,024 B to 8,192 B | `4N`, the fitness scalars | **2.00** |
| strategy B | 6.29 MB to 100.66 MB | one params-sized all-reduce, plus `4N` | 1.00 |

**A needs two all-gathers of `N` scalars, not one.** `tell` replicates the fitness so
`centered_ranks` can sort it globally, and `contraction.py` replicates ids and weights
because strategy A contracts the whole population on every device. Both gathers are visible
in the HLO, both `f32[N]`, both from `sharding_constraint`. They were free while the
evaluation was replicated, because there was nothing to gather; the sharding fix made them
real. So the folk claim that ES only all-reduces scalars remains true of A in order of
magnitude, and the constant is 2, not 1.

**The shaping barrier is no longer measurable by difference, and the `shaping` column now
reads 0 for the wrong reason.** `comms.py` compiles each configuration against
`shaping=none` and subtracts. `tell` replicates unconditionally, so `none` pays the same
gather, and the difference is 0 while the cost is `4N`. Before the sharding fix that 0 was
correct: the fitness was replicated anyway and the barrier genuinely added nothing. Now it
is an artifact of the method. Attributing it properly means either making `tell`'s replicate
conditional on the shaping, which `core.py` argues against, or measuring the barrier some
other way.

**Against the clock, communication is still negligible.** The pre-fix figure was 1.894 GB/s
at the most demanding configuration, 0.3% of an A100's ~600 GB/s NVLink. Post-fix the bytes
roughly double for A and are unchanged for B, while generations get faster (6.75 ms against
11.46 ms at `D=2` for `lowrank_r1`), so the requirement rises by something under 2x and
stays under 1% of the interconnect. That figure is an extrapolation rather than a
measurement: recomputing it exactly needs post-fix generation times for every configuration,
which means re-running the sweep. This is the measurement that says the flat scaling curve
was not a communication problem.

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

## Results, 2026-08-11, after the sharding fix

The 2026-08-06 section above is kept as written. It is the other half of a before and after,
not a draft to be corrected: every number in it was correctly measured, on a program whose
evaluation was replicated on every device. What changed is the program.

**The run.** The same 256 configurations, from `experiments/phase2/sweep-postfix.yaml`, which
differs from `sweep.yaml` by one line, `results_dir`. 0 failed, 0 over cap, 3h57m of measured
wall on 8x A100-SXM4-80GB (driver 595.71.05, CUDA 13.2, jax 0.11.0, community cloud).
`XLA_FLAGS` and every other knob are unchanged. Data in `experiments/phase2/results-postfix/`,
figures in `experiments/phase2/figures-postfix/`, commit `a496345`, all 256 records stamped
`dirty_worktree: false`.

Billed $52.43 against 4h43m of pod uptime. The gap between 3h57m of measurement and 4h43m of
billing is setup, two failed bootstraps and the tail after the sweep finished. Budget from
uptime, not from the wall clock the driver reports.

**Comparability is enforced, not asserted.** `experiments/phase2/compare.py` holds the
platform, chip, jax, jaxlib and `XLA_FLAGS` fixed across the two runs and allows only the
commit to differ, then checks every configuration that appears in both. **All 64 `D=1`
trajectory digests are bitwise identical to 2026-08-06.** `D=1` is where the fix is a no-op,
so that is the check with teeth: the two sweeps ran the same library on the same arithmetic,
and what follows is a property of the change rather than of the machine.

At `D>1` bitwise equality is neither expected nor demanded, and the reason is the defect
itself. Before the fix every device count ran the same program shape, a vmap over all `N`
members `D` times over, which is why `check.py` reported strategy A exact across
`D=1,2,4,8` for the whole pre-fix sweep. That pass was vacuous. After the fix `D=4` vmaps
over `N/4`, reduces in a different order, and lands a fraction of an ulp elsewhere. 22
multi-device updates moved, by 6.34e-05 to 8.22e-04. Four are adjudicated as the documented
noise floor by their rank digests; **18 cannot be adjudicated at all**, because the 2026-08-06
baseline predates `rank_digest` and does not record whether members changed rank. They are
reported as unadjudicated rather than counted as passes.

### M1, strong scaling: it works now

Parallel efficiency `T1/(D*T8)` at `D=8`, against **0.112 to 0.142 everywhere** before:

| | `D=8` efficiency, post-fix | before |
|---|---|---|
| `iid_gaussian / B` | 0.560 to **0.815** | 0.119 to 0.127 |
| `lowrank_r1 / A` | 0.382 to **0.728** | 0.112 to 0.124 |
| `mirrored_lr1 / A` | 0.379 to 0.704 | 0.112 to 0.124 |
| `iid_gaussian / A` | 0.313 to 0.367 | 0.119 to 0.125 |
| `seed_regenerated` | 0.123 to 0.142 | 0.124 to 0.142 |

Mean parallel efficiency over the 112 multi-device configurations present in both runs:
**0.298 to 0.621**. Excluding `seed_regenerated`, which is a separate defect and is treated
below, `D=8` efficiency spans **0.313 to 0.815**, a gain of 2.6x to 6.5x over the same
configuration measured on the same hardware.

Efficiency rises with problem size, 0.31 to 0.42 at the smallest block and 0.59 to 0.82 at
the largest, which is ordinary amortization of fixed per-generation cost and not a property
of the design.

**`seed_regenerated` does not distribute, and this is a second defect rather than a slow
strategy.** Its wall clock is flat in `D` (108.56, 109.04, 109.61 ms at `D=1,2,4` for
`d=512, N=256`), efficiency is `1/D` to three digits, and `profile.py --static` shows
per-device eval FLOPs unchanged from `D=1` to `D=8`. Its `apply` produces the evaluation with
`lax.scan` rather than `vmap`, and a scan's iteration space is a sequential loop that GSPMD
cannot partition, so the output constraint is satisfied by gathering the ids and running all
`n` iterations everywhere. `docs/diagnosis-seed-regenerated-scan.md` has the measurement and
the mechanism.

**Fixed on 2026-08-11, after this sweep ran, so every `seed_regenerated` row above and below
is stale.** `ShardedES.apply` now reshapes the member axis to `(D, n/D)` and vmaps over it,
which gives GSPMD a batch axis to partition instead of a loop it must decline.
`docs/proposal-scan-strategies-distribute.md` records the four options and why `shard_map`,
which also distributes, was rejected: it puts the user's model inside a manual mesh and
breaks three MuJoCo rollout tests. The obvious repair, replacing the scan with a vmap, was
never a candidate, since it would distribute the work and destroy the `O(|params|)` memory
that is the entire point of seed regeneration.

The other three strategies are unaffected: their evaluation already distributed and the
change does not alter what they compute. **Re-running the `seed_regenerated` configurations
is the outstanding measurement**, and until it happens M1, M2, M3 and M6 should be read as
describing a `seed_regenerated` that did not distribute.

### M2, weak scaling

Throughput at `D=8` as a multiple of `D=1`, ideal 8x: **1.00x to 7.53x, median 5.69x** across
32 series, against **1.01x to 1.58x, median 1.20x** before. Excluding `seed_regenerated`,
**3.01x to 7.53x, median 6.26x**. The best is `iid_gaussian/B` at `d=2048, N/device=32`.

### M3, the contraction crossover

`log10(t_B/t_A)` at `D=8` now spans **-0.373 to +0.059**, against -0.057 to +0.034 before,
and B is faster in 12 of 16 cells. The pre-fix reading, that the two contractions are within
14% everywhere, was a statement about a replicated evaluation dominating both.

| strategy | `log10(t_B/t_A)` | reading |
|---|---|---|
| `iid_gaussian` | -0.373 to -0.252 | B everywhere, 1.8x to 2.4x faster |
| `seed_regenerated` | -0.057 to -0.027 | B everywhere, but see below: this row is not a contraction result |
| `lowrank_r1` | -0.039 to +0.049 | mixed, sign flips with model size |
| `mirrored_lr1` | -0.004 to +0.059 | mixed, sign flips with model size |

**The split is explained by where A's cost sits, and the FLOP counts say so directly.** At
`d=2048, N=256` the evaluation distributes perfectly under both contractions, `evalFLOP`
falling to 0.125 at `D=8` against an ideal `1/8 = 0.125`. The whole generation does not:

```
                                      evalFLOP  fullFLOP
  d=2048 N=256 iid_gaussian/A D1->D8     0.125     1.109
  d=2048 N=256 iid_gaussian/B D1->D8     0.125     0.125
  d=2048 N=256 lowrank_r1/A   D1->D8     0.125     0.273
  d=2048 N=256 lowrank_r1/B   D1->D8     0.125     0.133
  d=2048 N=256 mirrored_lr1/A D1->D8     0.125     0.213
  d=2048 N=256 mirrored_lr1/B D1->D8     0.125     0.146
```

Strategy A regenerates and contracts the whole population on every device, which `docs/02`
C1.3 chose deliberately to trade compute for communication. Under B the whole generation
falls at 0.125, the ideal. Under A it does not fall at all for `iid_gaussian`, 1.109, because
regenerating `N` full-rank perturbations per device costs about what evaluating them does and
that cost is independent of `D`. For the low-rank strategies A's contraction runs over rank-1
factors and is far cheaper, 0.273 and 0.213, which is why `lowrank_r1/A` still reaches 0.728
measured efficiency while `iid_gaussian/A` stops at 0.367. That is the A/B tradeoff becoming
visible for the first time; before the fix nothing scaled and there was nothing to trade.

**These are GPU numbers, and the same table on CPU disagrees.** `profile.py --static` on 8
simulated CPU devices reports `iid_gaussian/A` at 0.260 rather than 1.109 and `lowrank_r1/A`
at 0.135 rather than 0.273. The evaluation column agrees, 0.125 on both. This document
previously carried the CPU figures on the assumption, stated in `profile.py`, that absolute
FLOP counts are backend dependent but ratios are portable. **That assumption is wrong for the
contraction**, which under A regenerates the same perturbation the evaluation already built,
leaving the two backends free to disagree about how much of it is common subexpression. Quote
the backend the sweep ran on, which is the table above.

**`seed_regenerated`'s row measures the wrong thing and should not be read as a contraction
result.** M3 is a ratio of whole generations, which is a statement about the contraction only
when the contraction is a meaningful share of the generation. For `seed_regenerated` the
evaluation is 399 ms of a 432 ms generation and does not distribute at all, so the ratio is
dominated by a part that is identical under both contractions and the contraction difference
is compressed toward zero. The per-part timings show what the ratio hides: at
`d=512, N=1024`, `tell` scales `D=1` to `D=8` at **0.15 under B and 1.02 under A**, which is
the difference between a contraction that shards and one that does not. The 3% to 6% in the
table is that difference divided by a generation the defect made four times too long. The
row will move once the strategy distributes.

Still no crossover contour. Two model sizes by three populations cannot carry one, and that
is unchanged by any of this.

### M5, communication against the clock

**Recomputed, not extrapolated, and the previous figure in this document was wrong.**
`experiments/phase2/bandwidth.py` divides the payload `comms.py` counts by the generation time
`run.py` measures. At the most demanding configuration, `d=2048, N=128, lowrank_r1/B` at
`D=8`, 100,663,808 bytes per generation at 11.75 ms is **8.564 GB/s, 1.43% of an A100's
~600 GB/s NVLink**.

The M5 section above estimated that post-fix the requirement would rise "by something under
2x" and stay "under 1% of NVLink". It rose 4.5x, from 1.894 GB/s, and 1.43% is not under 1%.
The estimate counted the bytes doubling for strategy A and underweighted the generations
getting faster: the same 100 MB all-reduce now lands in 11.75 ms instead of roughly 50. The
conclusion survives, communication is still a rounding error against NVLink, but the number
in an estimate is not a measurement and this document should not have carried one.

### M6, memory: weak scaling now actually holds per-device memory constant

**28 of 32 weak-mode series are flat in `D`.** That is what weak scaling is supposed to look
like: population per device held constant, memory per device constant.

| weak, `d=512, N/device=128` | `D=1` | `D=2` | `D=4` | `D=8` |
|---|---|---|---|---|
| `iid_gaussian/B` before | 1610 | 1674 | 3338 | **6666** MiB |
| `iid_gaussian/B` after | 1610 | 1610 | 1610 | **1610** MiB |
| `lowrank_r1/B` before | 206 | 395 | 1031 | **1554** MiB |
| `lowrank_r1/B` after | 206 | 205 | 205 | **205** MiB |

Before the fix, adding devices at fixed per-device population grew per-device memory 4x to
7.5x, because every device held the whole population. That is not weak scaling; it is a
configuration that runs out of HBM by adding hardware. The pre-fix M6 section reported A's
storage tracking total `N` as a designed property, which it is, and did not notice that B was
doing the same thing for a reason that was not.

Three series still grow, all `iid_gaussian/A`, as designed. `seed_regenerated` is flat at 15
MiB (`d=512`) and 144 MiB (`d=2048`) across every device count, still the clearest systems
result in the sweep. **Unlike its M1 and M3 rows, this one is not distorted by the defect
above.** Its scan holds one member at a time, so per-device storage is `O(|params|)`
whatever `n` and whatever `D`, and that stays true once the evaluation distributes. The
memory claim is about the schedule, and the defect is about where the schedule runs.

One anomaly survives unchanged: `iid_gaussian/A` at `d=512, N/device=32` reads 486, 454, 902,
1798 MiB, dipping 6.6% at `D=2` before doubling. The pre-fix run had the same shape (486,
426) and the same non-explanation. It is the step where the program goes from unsharded to
sharded, a different compiler choice is still the obvious suspect, and it has still not been
checked.

### What did not need redoing, and why

`noisefloor.txt` stands. `noisefloor.py` builds its mesh with `sharding.make_mesh(1)` and is
a single-device measurement by construction, and all 64 `D=1` digests are bitwise identical
across the fix. A `D=1` measurement whose inputs did not move cannot have moved.

`comms.txt` was already re-run post-fix on 2026-08-07. This run confirms it: `comms.py`
against `sweep-postfix.yaml` reproduces the same payloads, A at 1,024 to 8,192 bytes and B at
6.29 to 100.66 MB.

### Two defects in the instrumentation, found by re-running

**`plot.py` drew M1 from a quarter of the data.** The M1 series keyed on `(strategy, how)`
with neither model size nor population, so the four blocks collapsed onto every line and the
last by filename order won each device count. That was `d=512, N=256` consistently, so the
ratios stayed inside one block by luck rather than by construction, and the committed figure
showed a quarter of the sweep captioned as all of it. It also happened to show the worst
block: at `d=512, N=1024` efficiency reaches 0.63 to 0.78 at `D=8` against 0.31 to 0.56 at
`N=256`. M1 is now faceted like M2 and M3. **This is the third time this file has had a
series key missing a factor of the design**, which is worth stating plainly: a dict key that
drops a dimension produces a plausible line, not an error.

**`compare.py` first demanded bitwise equality at every device count**, which demands that
the fix not have happened. Corrected to assert at `D=1` and report at `D>1`.

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
| 1 | strong and weak curves, efficiency stated | **met**, cause identified and since fixed |
| 2 | crossover measured, phase diagram | **met**, without a contour: the grid is too coarse to carry one |
| 3 | one comparison against an external reference | **not met**, M4 was not run |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | drafted above, needs rewriting |

**The cause has since been fixed, and the sweep's numbers are now history rather than
current behaviour.** `ShardedES.apply` constrains its output to the member axis, which is
what forces the evaluation to partition. Measured on 2x A100 with both code versions on one
node (`experiments/phase2/wallclock.txt`), parallel efficiency at `D=2` for `lowrank_r1`
goes from **0.48 to 0.70** under strategy A and **0.47 to 0.85** under B. Pre-fix `D=2` was
slower than `D=1`; post-fix the wall clock drops and `eval` falls to 0.56 against an ideal
0.50. `docs/diagnosis-replicated-evaluation.md` carries the account.

Everything in this section describes the run of 2026-08-06 and stays valid as a record of
it. **Re-running the sweep would now produce a materially different scaling curve, and that
is a new measurement rather than a correction.**

**Criterion 1's clause is now satisfied.** Scaling came in at 0.11 to 0.14 efficiency, far
worse than expected, and the gate forgives that only when the cause is identified and
measured. It is: per-device eval FLOPs are unchanged from `D=1` to `D=8` in all 16
configurations, so every device evaluates the whole population and `1/D` efficiency is the
arithmetic consequence. "ES scales at 71% and here is where the other 29% goes" was the
standard this section set; "it scales at 1/D because the evaluation is replicated rather
than sharded" meets it, even though the answer is worse than hoped.

Three measurements agree and are worth reading together: the evaluation does not distribute
(profile), the contraction barely matters (M3, within 14%), and communication is negligible
(M5, 0.3% of NVLink). The last two are consequences of the first.

**The gate still does not pass, on criterion 3.** M4 has not been run, so there is no
comparison against an external reference. That is now the only outstanding measurement.

Whether to fix the sharding before or after M4 is a judgement call, not a gate question. A
scaling curve measured against EGGROLL while the evaluation is replicated `D` times
measures the defect, not the design.

### Status after the 2026-08-11 re-run

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met**, and the curves now show scaling rather than its absence |
| 2 | crossover measured, phase diagram | **met**, still without a contour, but the diagram now has structure to show |
| 3 | one comparison against an external reference | **not met**, M4 still not run |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | still needs rewriting, in Andres's words |

**Criterion 1 is met on its own terms now, not on the forgiveness clause.** Strong scaling
reaches 0.313 to 0.815 parallel efficiency at `D=8` across the three strategies that
distribute, and where it falls short the cause is measured rather than guessed: strategy A's
contraction caps `iid_gaussian` at 0.260 full-generation FLOP scaling, and fixed
per-generation cost dominates the smallest block. "ES scales at 71% efficiency to 8 GPUs and
here is where the other 29% goes" was the standard this document set for itself. That is now
roughly the literal result.

**One defect is outstanding and it is not a gate question either.** `seed_regenerated` does
not distribute its evaluation at all, for a reason unrelated to the fix that made the others
work: `lax.scan` cannot be partitioned by GSPMD. It is measured, diagnosed and unfixed
(`docs/diagnosis-seed-regenerated-scan.md`). Both published algorithms this library exists to
support are affected, since `Mirrored(SeedRegenerated())` is Qiu et al.

**The gate still does not pass, on criterion 3.** M4 remains the only outstanding
measurement, and it is now worth running for the right reason: benchmarking against EGGROLL
while the evaluation was replicated `D` times would have measured the defect.
