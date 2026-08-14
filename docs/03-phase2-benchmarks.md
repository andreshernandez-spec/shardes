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

> **Superseded as a description of current behaviour by "Results, 2026-08-14", and kept as
> the before half of a before and after.** Every number here was correctly measured
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
`experiments/phase2/figures-history/2026-08-06-prefix/`, guard output from `check.py`.

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

> **Superseded by "Results, 2026-08-14" below, which measures all four strategies at one
> commit. The `iid_gaussian`, `lowrank_r1` and `mirrored_lr1` numbers here are absolute
> figures for a program that no longer ships.** They were measured at `a496345`, before
> `ShardedES.apply` reshaped the member axis. That change re-derives the perturbation inside
> the vmap, which is a third materialisation, so current `main` does 1.32x the FLOPs for
> `iid_gaussian` and about 1.08x for the low-rank strategies, with peak memory up 10 to 18%.
> Measured; see the addendum in `docs/proposal-scan-strategies-distribute.md`.
>
> **Scaling ratios survive**, which is what M1 efficiency and M2 throughput rest on: the
> `D1->D8` FLOP ratio for `iid_gaussian` is 0.1260 against 0.1262. **Absolute ms/generation
> and MiB/device do not.** The `seed_regenerated` rows were re-measured on 2026-08-13 and are
> current; M4 is current. Putting all four strategies on one commit is 192 configurations and
> about $30, and has not been done.


The 2026-08-06 section above is kept as written. It is the other half of a before and after,
not a draft to be corrected: every number in it was correctly measured, on a program whose
evaluation was replicated on every device. What changed is the program.

**The run.** The same 256 configurations, from `experiments/phase2/sweep-postfix.yaml`, which
differs from `sweep.yaml` by one line, `results_dir`. 0 failed, 0 over cap, 3h57m of measured
wall on 8x A100-SXM4-80GB (driver 595.71.05, CUDA 13.2, jax 0.11.0, community cloud).
`XLA_FLAGS` and every other knob are unchanged. Data in `experiments/phase2/results-postfix/`,
figures in `experiments/phase2/figures-history/2026-08-11-postfix/`, commit `a496345`, all 256 records stamped
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

## Results, 2026-08-13: seed_regenerated re-measured, and M4

Two runs on one 8x A100-SXM4-80GB node (driver 595.71.05, CUDA 13.2, jax 0.11.0, community
cloud, commit `5c18799`, 1 h, $11). All 64 re-run records stamp `dirty_worktree: false`.

### The seed_regenerated rows above are superseded

`experiments/phase2/sweep-seedregen.yaml` re-runs the 64 `seed_regenerated` configurations
after `ShardedES.apply` began reshaping the member axis into a batch axis
(`docs/proposal-scan-strategies-distribute.md`). It differs from `sweep-postfix.yaml` by two
lines, `strategies` and `results_dir`. The other three strategies are not re-run: their
evaluation already distributed and the reshape does not change what they compute.

**It distributes now.** Wall clock at `d=512, N=256`:

| | `D=1` | `D=2` | `D=4` | `D=8` |
|---|---|---|---|---|
| before | 108.6 ms | 105.4 | 104.2 | **103.6** |
| after | 108.6 ms | 58.7 | 37.4 | **24.1** |
| efficiency after | 1.000 | 0.924 | 0.726 | **0.563** |

Merged with the three strategies that were already current, the sweep now reads:

- **M1**: parallel efficiency at `D=8` spans **0.313** (`iid_gaussian/A`) to **0.951**
  (`seed_regenerated/B`). `seed_regenerated` alone went from 0.123-0.142, which is `1/D`, to
  **0.464-0.951**.
- **M2**: weak throughput at `D=8` is **3.01x to 7.53x** of one device against an ideal 8x,
  median **6.26x**. `seed_regenerated` went from 1.00x to **4.01x-7.43x**.
- **M3**: `seed_regenerated`'s cells moved from `-0.027..-0.057` to **`-0.159..-0.275`**, so
  B is 1.4x to 1.9x faster than A rather than the 3-6% previously reported. The earlier
  section flagged that row as "not a contraction result" because a 399 ms non-distributing
  evaluation was drowning the signal in a 432 ms generation. It was, and this is the size of
  what it was hiding.
- **M6**: unchanged at 15 MiB (`d=512`) and 144 MiB (`d=2048`) per device. Also flagged
  earlier as the one `seed_regenerated` row the defect did *not* distort, because the scan
  holds one member at a time wherever it runs. That held.

**The fix is numerically inert here, at every device count.** `compare.py` reports 16/16
`D=1` digests exact and **zero** multi-device updates moved, which is stronger than the
`apply` fix managed for the low-rank strategies. The reason is structural: this evaluation
has no cross-member reduction. The scan computes each member's loss independently and stacks
them, so 64 members on one device and 8 on each of eight produce bitwise identical fitness.
The reshape changed only where the iterations run.

`check.py` on the new results: all 8 strong groups invariant across `D=1,2,4,8`, strategy A
bitwise exact and B at 2.7e-07 to 4.5e-07 from summation order, **no noise-floor groups**.
Unlike the low-rank strategies, `seed_regenerated`'s populations are not packed inside
float32 resolution at these shapes.

### M4, against the references

`experiments/phase2/m4.py`, tokens/s at a stated parameter count, three arms on the same
model, population and batch. `tokens = population * batch * seq`.

**Like for like, one A100, which is what `docs/03` asks for:**

| `d=512, N=1024`, 1 GPU | ms/gen | tokens/s |
|---|---|---|
| shardes `mirrored_lr1/B` | **9.84** | **26.6M** |
| evosax `Open_ES` | 45.03 | 5.8M |
| naive ES | 46.23 | 5.7M |
| shardes `seed_regenerated/B` | 179.84 | 1.5M |

The low-rank path is **3.9x to 9.0x faster than evosax** across the grid, and the gap widens
with model size, which is what a factored perturbation predicts: naive cost scales with
`n*|params|` and low rank with `n*r*(m+k)`.

**evosax and the naive baseline agree to within 2% everywhere** (12.04 against 12.11 ms,
45.03 against 46.23, 173.02 against 160.65). That was not arranged. It says the incumbent and
"the baseline both papers beat" are the same measurement, which is the concrete form of the
`ravel_pytree` argument in `docs/00`, and it means the naive arm is not a strawman built to
lose.

**`seed_regenerated` is 2.6x to 4x slower than naive on one device**, and that is the
strategy working as designed rather than a defect. It buys `O(|params|)` storage with
throughput, which M6 prices at 15 MiB per device against 12,810 for `iid_gaussian/A`.

**With the mesh, which is not like for like:** at `d=2048, N=256` shardes `mirrored_lr1/B`
reaches 10.9M tokens/s against evosax's 0.379M, a factor of 28.7. That decomposes into
**9.0x algorithmic**, measured at `D=1` on the same GPU, and **3.2x from sharding** across
eight devices, consistent with M1's efficiency. evosax and the naive arm have no sharding
path, so they run on one device whatever the mesh says; `m4.py` prints that caveat above the
table rather than leaving the ratio to be read as a library-against-library number.

The most interesting row is `seed_regenerated`: **4x slower than evosax at `D=1` and 1.6x
faster at `D=8`.** Trading throughput for memory on one device and winning it back through
device count is Qiu et al.'s argument, and this is it measured.

### What M4 does not cover

**EGGROLL's own implementation was not run.** `ESHyperscale/HyperscaleES` is GPL-3.0 against
this repo's Apache-2.0, so none of it is vendored and none should be; `m4.py` imports it if
the person running the benchmark installed it, exactly as it treats evosax. Beyond the
licence, their API is built around RWKV language models, so driving it at this transformer
block is adaptation work rather than an import. `docs/03` calls that the comparison that
matters, and it is still outstanding.

**evosax did not run out of memory**, including at `d=2048, N=256` where its flattened
population is about 26 GB. It fit in 80 GB and ran. So M4 measures a throughput gap and not
the memory wall; that argument still rests on M6.

---

## Results, 2026-08-14: one commit, all 256 configurations

**This section supersedes the three above as a description of current behaviour.** They stay
as the record of what was measured when, and the sequence is the point: 2026-08-06 measured a
replicated evaluation, 2026-08-11 measured three strategies after the first fix,
2026-08-13 measured the fourth after the second. Every one of those tables mixes commits with
the others. This one does not.

**The run.** 256 configurations from `sweep-consistent.yaml`, which differs from `sweep.yaml`
by one line, `results_dir`. One 8x A100-SXM4-80GB node (driver 595.71.05, CUDA 13.2, jax
0.11.0, community cloud), commit `5769751`, 4h09m, $46. **256 written, 0 failed, 0 over cap,
0 needed more devices**, and every record stamps `dirty_worktree: false` at a single commit.

M4 and `profile.py` ran in the same session on the same node, so the comparison arms and the
per-part breakdown are the same program as the sweep. `experiments/phase2/results-consistent/`,
`results-m4-consistent/`, `figures-history/2026-08-14-consistent/`, `profile-consistent.txt`.

### M1, strong scaling

Parallel efficiency `T1/(D*T8)` at `D=8`, **0.292 to 0.939**:

| strategy | `D=8` efficiency | at 2026-08-06 |
|---|---|---|
| `seed_regenerated` | **0.470 to 0.939** | 0.124 to 0.142 |
| `iid_gaussian` | 0.292 to 0.750 | 0.119 to 0.127 |
| `lowrank_r1` | 0.428 to 0.673 | 0.112 to 0.124 |
| `mirrored_lr1` | 0.395 to 0.662 | 0.112 to 0.124 |

**`seed_regenerated` is now the best scaler, and it is the strategy that spent two sweeps
pinned at `1/D`.** That is not a coincidence: its evaluation is a `lax.scan` over members, so
it has the least per-generation overhead to amortise once the scan is actually divided. The
worst is `iid_gaussian/A` at 0.292, and `profile.py --static` says why: its `evalFLOP` falls to
0.125, the ideal, while `fullFLOP` stays at 1.109, because strategy A regenerates and
contracts the whole population on every device and for a full-rank perturbation that costs
about what evaluating it does.

### M2, weak scaling

Throughput at `D=8` as a multiple of `D=1`, ideal 8x: **3.00x to 7.61x, median 6.32x**,
against 1.01x to 1.58x, median 1.20x on 2026-08-06.

### M3, the contraction crossover

`log10(t_B/t_A)` at `D=8` spans **-0.378 to +0.059**, and B is faster in 10 of 16 cells. The
2026-08-06 reading of "within 14% everywhere" was a statement about a replicated evaluation
dominating both contractions rather than about the contractions.

### M6, memory: and this is where the story changed

**Two things happened at once, and they need separating.**

**First, the measurement was wrong for every previous run.** `run.py` compiled its memory
analysis outside the configured `matmul_precision` context, so it measured a DEFAULT-precision
program while the timings ran at `highest`, and it summed `temp + argument` while omitting
`output - alias`. The missing term is the parameters the updated state has to hold: a constant
**6 MiB at `d=512` and 96 MiB at `d=2048`**. Constant, so negligible for a 3 GiB strategy and
**41% for a 15 MiB one**, which is exactly the number this section used to lead with.
`seed_regenerated` at `d=512` reads 21 MiB rather than 15, and 240 rather than 144 at
`d=2048`. The flagship memory claim was understated, so correcting it makes the result
slightly less impressive rather than more.

**Second, strategy A's per-device memory now falls with the device count.** Measured, strong
mode, peak GiB per device:

| `iid_gaussian` | `D=1` | `D=8` before | `D=8` now |
|---|---|---|---|
| `d=2048, N=256, /A` | 48.69 | 28.09 | **3.25** |
| `d=2048, N=256, /B` | 48.69 | 6.16 | 6.25 |
| `d=512, N=1024, /A` | 12.52 | 7.01 | **0.83** |
| `d=512, N=1024, /B` | 12.52 | 1.57 | 1.58 |

**A is now cheaper than B at `D=8`, where it used to be 4.6x more expensive.** In weak mode
the same thing: `iid_gaussian/A` at `d=512, N/device=128` read 1610, 1798, 3590, 7174 MiB and
now reads 1616, 848, 848, 848. 28 of 32 weak-mode series are flat in `D`.

This contradicts what this document said as recently as 2026-08-11: "Strategy A storage does
not fall with `D` in weak mode, as designed: every device regenerates the whole population, so
per-device storage tracks total `N`." **That is no longer true and the change is in the code,
not the measurement**: the `output - alias` correction *adds* memory uniformly and cannot
produce an 8.6x reduction. `ShardedES.apply` now reuses the perturbation `ask` built rather
than re-deriving it per row (`docs/proposal-scan-strategies-distribute.md`), and XLA fuses the
contraction's regeneration into its reduction instead of materialising the full `(n, ...)`
perturbation first.

**It is correct, not a shortcut.** Strategies A and B agree on the update for **all 128
configurations** at `rtol=1e-5`, worst disagreement 4.54e-07, which is summation order.

`D=1` is unchanged at 48.69 GiB, so `feasible.py`'s model and the `N=256` ceiling at `d=2048`
still stand: the ceiling is set at one device, where there is nothing to divide.

### M5, communication against the clock

**7.928 GB/s, 1.32% of an A100's ~600 GB/s NVLink**, at the most demanding configuration
(`d=2048, N=128, lowrank_r1/B` at `D=8`). Both halves now come from the same commit:
`comms.py` counts the payload in the compiled HLO and `run.py` times the generation, and
`bandwidth.py` divides. Previously the bytes and the times were from different programs.

The collective structure is unchanged from every earlier run, verified: same ops, same
payloads, A at 1,024 to 8,192 bytes and B at 6.29 to 100.66 MB. Communication has never been
the constraint and still is not.

### M4, against the references

Like for like on one A100, which is what this document asks for:

| `d=512, N=1024`, 1 GPU | ms/gen | tokens/s |
|---|---|---|
| shardes `mirrored_lr1/B` | **10.26** | **25.6M** |
| evosax `Open_ES` | 45.04 | 5.8M |
| naive ES | 45.24 | 5.8M |
| shardes `seed_regenerated/B` | 182.90 | 1.4M |

The low-rank path is **3.7x to 8.4x faster than evosax** across the grid, widening with model
size, which is what a factored perturbation predicts.

**evosax and the naive baseline agree to within 2% at `d=512` and within 8% at `d=2048`**,
where naive is the faster of the two. An earlier revision of this document said "within 2%
everywhere", which was true of the shapes it had looked at. The point survives: the incumbent
and the baseline both papers beat are the same measurement, which is the concrete form of the
`ravel_pytree` argument in `docs/00`.

With the mesh, which is **not** like for like: at `d=2048, N=256` shardes reaches 10.3M
tokens/s against evosax's 0.377M, a factor of 27.4. That decomposes into 8.4x algorithmic,
measured at `D=1` on the same GPU, and 3.3x from sharding across eight devices. evosax and the
naive arm have no sharding path, so they run on one device whatever the mesh says, and `m4.py`
prints that caveat above the table.

`seed_regenerated` is 4x slower than evosax at `D=1` and **3.1x faster at `D=8`**. Trading
throughput for memory on one device and winning it back through device count is Qiu et al.'s
argument, measured.

### What the guards said

`check.py --results results-consistent --config sweep-consistent.yaml` returns 1, and the
reason is the noise floor rather than anything about this run: **6 groups reorder their
population across device counts**, all at `d=512`, the same ones every previous sweep flagged.

    mirrored_lr1/A and /B  d=512 N=256   (1.45e-03)
    lowrank_r1/A and /B    d=512 N=1024  (1.09e-04)
    mirrored_lr1/A and /B  d=512 N=1024  (1.17e-04)

Their populations are packed inside float32 resolution, so an ulp of arithmetic reorders them
and the rank shaping turns that into a different update. It is a property of those
configurations. **No scaling number above should be quoted for those six groups without
saying so.**

There were **no errors and no missing rows**: the matrix validation found all 256 expected
configurations present. Those are the two conditions that would have invalidated the run, and
both are clean.

### Two defects in the harness, found by running it

**`check.py` returned 1 for three different things**: recorded errors, missing rows, and the
noise floor. That makes it useless as a mid-session gate, because the noise floor is permanent
for six `d=512` shapes, so a session gating on it would never proceed and a session ignoring
it would ignore the errors too. Now **2 means the run is broken** (errors or holes), **1 means
complete but some group's scaling number would mislead**, and 0 means nothing to report. A
caller renting hardware should stop on 2 and carry on past 1.

**And the session driver was not gating at all.** The script that ran this sweep printed
`check rc=1` between the sweep and M4 without branching on it, so a sweep that had errored
would still have funded the rest of the session. It was harmless here only because the sweep
was clean. That script lives with the runbook rather than in the repository, so the durable
half of the fix is the exit code above; the driver has to be the thing that reads it.

---

## Results, 2026-08-14b: the Qiu configuration, measured at last

`Mirrored(SeedRegenerated())` is Qiu et al. as published, and `core.py` has advertised it by
name since Phase 1:

    ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, ...)          # Qiu et al.

No sweep had ever run it. `sweep.yaml` measures `seed_regenerated` unmirrored and
`mirrored_lr1` for the low-rank half, so the published pairing of antithetic sampling with
seed regeneration was the one cell of the design with correctness tests and no performance
numbers. It is also the configuration that inherited the scan defect and was fixed without
being measured.

**The run.** 64 configurations from `sweep-qiu.yaml`, which differs from `sweep.yaml` by
`strategies` and `results_dir`. Same node type, commit `eee4bd1`, ~50 min, ~$10. **64 written,
0 failed, 0 over cap**, single commit, no dirty worktrees.

| | Qiu, `mirrored_seed` | unmirrored `seed_regenerated` |
|---|---|---|
| M1 efficiency at `D=8` | **0.567 to 0.931** | 0.470 to 0.939 |
| M2 weak `D=8/D=1` | **5.12x to 7.68x**, median 6.77x | median 6.32x across all four |
| M3 `log10(t_B/t_A)` at `D=8` | -0.154 to -0.083, B faster in 4/4 | -0.057 to -0.027 |
| M6 weak, MiB per device | 21 and 240, flat in `D` | 21 and 240, flat in `D` |

**It is faster than the unmirrored strategy in 7 of 8 cells at matched `N`, and the
advantage lives almost entirely in contraction A.** Wall clock at `D=8`:

| shape | how | unmirrored | Qiu | speedup |
|---|---|---|---|---|
| `d=512, N=256` | A | 26.5 ms | 19.8 | **1.34x** |
| `d=512, N=1024` | A | 83.6 | 67.8 | **1.23x** |
| `d=2048, N=128` | A | 59.0 | 45.5 | **1.30x** |
| `d=2048, N=256` | A | 113.3 | 82.0 | **1.38x** |
| `d=512, N=256` | B | 18.3 | 16.3 | 1.12x |
| `d=512, N=1024` | B | 57.7 | 56.0 | 1.03x |
| `d=2048, N=128` | B | 37.0 | 31.9 | 1.16x |
| `d=2048, N=256` | B | 59.0 | 60.0 | 0.98x |

The saving is in the contraction rather than the evaluation. `Mirrored` draws `n/2` distinct
directions and evaluates each at plus and minus sigma, and its `contract` folds the signs into
the weights before delegating:

    sum_i w_i eps_i  with eps_2k = +e_k and eps_2k+1 = -e_k
      = sum_k (w_2k - w_2k+1) e_k

an exact identity rather than an approximation, valid for any inner strategy because
`contract` is linear. Verified numerically to 2.4e-07, which is float32 rounding. `tell` then
runs **half the scan iterations**, and for `seed_regenerated`, whose contraction *is* a
sequential scan rather than a GEMM, halving the iteration count is where the saving lands.

**Why A gains 1.23x to 1.38x and B gains almost nothing.** Under A every device regenerates
and contracts the whole population, so halving the directions halves a full-population scan on
every device. Under B each device contracts only its own `n/D` shard, so the same halving is a
much smaller absolute saving and disappears into other costs; at `d=2048, N=256` it is 2%
slower. The antithetic identity pays in proportion to how much contraction work is replicated,
which is a statement about the interaction of two design choices rather than about either one.

**The two arms are not sampling the same thing, so this is not "the same computation, done
faster".** At `N=256` the unmirrored strategy explores 256 independent directions and the
mirrored one explores 128, each evaluated twice. Both perform 256 model evaluations; only the
mirrored one halves the contraction, and it does so *because* it has half as many directions
to contract. Whether 128 antithetic pairs estimate the gradient better or worse than 256
independent draws is a variance question, it belongs to Phase 0
(`docs/01-phase0-estimator-harness.md`), and nothing in M1 measures it. Read these rows as
throughput at fixed `N`, not as a verdict on which sampling scheme is better.

Parallel *efficiency* does not follow the wall clock here, and the reason is worth one line:
under B at `d=2048, N=256` the unmirrored arm reads 0.939 against Qiu's 0.858 while being
slower in absolute terms, because efficiency is measured against each arm's own `D=1`
baseline and Qiu's is already faster. Efficiency compares an arm to itself; the table above
compares the arms to each other.

**M6 is unchanged from unmirrored**, 21 MiB at `d=512` and 240 at `d=2048`, flat in `D`. That
is the expected result and worth stating: pairing halves the distinct *directions*, not the
storage, which was already `O(|params|)` because the noise is regenerated rather than kept.

**The figures cover all five strategies.** `experiments/phase2/figures/` is plotted from
the union of `results-consistent` and `results-qiu`, 320 results, and those two directories
are `plot.py`'s defaults, so this is the no-argument invocation:

    python plot.py

Superseded sets moved to `experiments/phase2/figures-history/`, one dated directory per run,
with a README saying what each was and why it was replaced. They used to sit beside the
current one as `figures-postfix/`, `figures-consistent/` and so on, where nothing in the
names said which to read.

`plot.py` takes several directories rather than requiring a combined one, so there is no third
copy of every result on disk to go stale. Combining them is legitimate
rather than the stitching this document has spent three sections warning about: the two runs
are at `5769751` and `eee4bd1`, and **`git diff 5769751 eee4bd1 -- src` is empty**. Everything
between those commits is results, documentation and driver tables, so the library that
produced both sets of numbers is byte-identical. Each result records its own commit regardless.

**`check.py` returns 0, which no previous sweep has managed.** No errors, no missing rows, and
**no noise-floor groups**. Every earlier run flagged six `d=512` groups whose populations pack
inside float32 resolution so that an ulp reorders them; `mirrored_seed` is not among them,
because full-rank seed-regenerated perturbations spread the population far enough apart that
the rank transform has nothing to amplify. The low-rank strategies are the ones that crowd.

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


### Status after the 2026-08-13 runs

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met**, 0.313 to 0.951 at `D=8` |
| 2 | crossover measured, phase diagram | **met**, still no contour: the grid is too coarse |
| 3 | one comparison against an external reference | **met**, evosax at matched shapes |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | still needs rewriting, in Andres's words |

**G2's measurements are complete.** Criterion 3 is satisfied by evosax, which is a fair
external reference: it is the incumbent, it is what `docs/00` argues against, and it was run
on the same GPU at the same shapes with the same population.

It is satisfied in the letter and not in the spirit. The comparison `docs/03` asks for by
name is EGGROLL's own implementation, and that is not run. Being within its throughput while
offering a general API was set as a perfectly good result; we do not know whether we are.

**Criterion 5 is the remaining gate item**, and it is prose rather than measurement.


### Status after the 2026-08-14 run

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met**, 0.292 to 0.939 at `D=8`, one commit |
| 2 | crossover measured, phase diagram | **met**, no contour: the grid is too coarse to carry one |
| 3 | one comparison against an external reference | **met** by evosax, in the letter |
| 4 | reproducible from a committed config plus a recorded environment | **met**, and now from a *single* commit |
| 5 | a limitations paragraph a skeptic would accept as fair | **still needs rewriting, in Andres's words** |

**Criterion 4 is the one that changed.** It was met before in the sense that every number had
a config and an environment behind it. It was not met in the sense a skeptic means: the tables
mixed three commits, and two of the three had absolute figures for programs that no longer
existed. They do not now.

**Criterion 5 is the only gate item left**, and it is prose rather than measurement. The
material for it is unusually good: this phase found a replicated evaluation, a scan that could
not be partitioned, a memory measurement that was wrong for every run, a plotting bug that
showed a quarter of the data, and a checker that passed sweeps that had failed. A limitations
section that says so is stronger than one that does not.


## The EGGROLL arm, 2026-08-14

Criterion 3 has been met "in the letter and not in the spirit" since the first sweep: evosax
is a fair external reference, but the comparison this document asks for by name is EGGROLL's
own implementation, and the question it was written to answer was whether this library is
within their throughput while offering a general API. **It is: on one GPU the two are within
run-to-run variation of each other, across two runs and four shapes.**

### There was never an adapter to write

The arm was stubbed for a year on the belief that their library targets RWKV language models
and would need porting. That was wrong, and reading `src/hyperscalees/noiser/base_noiser.py`
is enough to see it. Their `Noiser` is model-agnostic:

| theirs | here |
|---|---|
| `do_mm(..., param, key, iterinfo, x)` | `dense(x, w)` |
| `do_Tmm(...)` | `dense` with the transpose the other way |
| `do_emb(...)` | `embed(table, ids)` |

Both libraries reached the same design: route every parameter read through a seam, and a
factored perturbation becomes expressible without touching the model. Theirs takes the noiser
as an argument, this one substitutes a structured weight into the params tree, and that is the
whole difference. `do_mm` is `x @ param.T` plus a rank-`r` correction, the same transpose
convention as `dense`.

So the arm is **their `EggRoll`, unmodified**, driven by a forward pass written here that
routes this project's transformer block through their seams. Six call sites. Not a port, and
nothing that has to track their model code. Their `do_emb` raises `NotImplementedError`, which
costs nothing here because this block has no embedding, and which is the case `shardes.nn`
already cites as the reason `embed` is a separate seam.

`ESHyperscale/HyperscaleES` at `b77f7d6`, GPL-3.0. **Not one line is copied**, and the arm
skips cleanly when the package is absent, which is the normal state. Vendoring it into an
Apache-2.0 repository would relicense the result and a benchmark is not worth that.

### The install is the awkward part, and the workaround is a judgement call

`hyperscalees/__init__.py` imports their model zoo, which needs gymnax, distrax, transformers,
datasets, torch and more. The `noiser` package this benchmark exercises imports `jax`, `optax`
and stdlib, and nothing else: checked across every file in `src/hyperscalees/noiser/`.

`_import_eggroll` tries the ordinary import first and uses it when it works. Otherwise it
loads their two noiser modules from the installed package directory, with a package skeleton
registered so their relative import resolves. Their code runs unmodified; only the route to it
skips `__init__`.

**This has two defensible answers and the other one was rejected for a stated reason.**
Installing their full dependency set puts torch's CUDA stack beside JAX's on the benchmark
node, and a throughput measurement is the wrong place to discover whether those coexist. The
cost is that this reaches past a package boundary and breaks if they move the file. It breaks
loudly, and there is a test for it. Worth revisiting if the arm ever runs anywhere it matters.

    git clone https://github.com/ESHyperscale/HyperscaleES.git   # b77f7d6, GPL-3.0
    pip install -e HyperscaleES --no-deps                        # never in pyproject.toml

### Three choices decide whether the comparison is fair

None of them is visible in the number, and two were nearly wrong.

**`rank=1`,** matching `LowRank(r=1)`, with `es_map` marking all six matrices `MM_PARAM` so
every one takes their low-rank path. Nothing frozen, nothing falling back to a dense
perturbation.

**`noise_reuse=1`.** Their default is `0`, and `0` means *reuse forever*, not "no reuse":
`true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse`. A default-constructed noiser
evaluates the same perturbations every generation. Their own experiment scripts pass the flag.
Timing barely moves either way, but a reader assumes fresh sampling and would be wrong.

**Their scheme is antithetic by construction**, `thread_id // 2` with the sign from
`thread_id % 2`. `N` EGGROLL members are `N/2` directions. **So the matched arm is
`mirrored_lr1`, not `lowrank_r1`**, and pairing it against the unmirrored arm would have let a
factor of two in sampling be reported as throughput. This is asserted in
`tests/test_m4_eggroll.py` rather than trusted from reading their source, along with the
rank of the correction and the fact that a perturbation is applied at all. A silently
unperturbed arm removes work, so it would surface as a throughput win rather than as an error.

### The result, one GPU

Two runs, `python m4.py --config sweep-consistent.yaml --devices 1 --out results-m4-local`
and `--out results-m4-local-2`, commits `a668b15` and `b805e44`, every record
`dirty_worktree: false`. **RTX 3080 Laptop, jax 0.11.0.** ms/gen, run 1 / run 2:

| shape | shardes `mirrored_lr1/B` | EGGROLL `rank1` | ratio | naive ES | evosax |
|---|---|---|---|---|---|
| `d=512, N=256` | 11.11 / 12.26 | 11.02 / 10.89 | 1.008 / 1.125 | 29.30 | 38.75 |
| `d=512, N=1024` | 42.64 / 41.15 | 42.07 / 41.60 | 1.014 / 0.989 | 118.69 | OOM |
| `d=2048, N=128` | 54.48 / 55.51 | 56.83 / 56.00 | 0.959 / 0.991 | OOM | OOM |
| `d=2048, N=256` | 108.29 / 106.10 | 107.08 / 106.48 | 1.011 / 0.996 | OOM | OOM |

**Parity, and it is reported as two runs because one run would have overstated it.** The
second run is here for a reason: a single run put the arms within 2.6% and reading that as
precision would have been wrong. Across both, the ratio ranges 0.959 to 1.125, and which arm
leads changes between runs at three of the four shapes.

The variation is not uniform. At `d=2048` the two runs agree to about 3% and the arms agree to
about 4%. At `d=512, N=256`, the smallest and shortest configuration, the same arm moves 10%
between runs, which is more than the gap being measured. That shape is where a laptop GPU's
clocks and a 11 ms generation stop being a measuring instrument.

So the defensible statement is the one that was set as the target: **a general, sharded API
costs nothing measurable against a specialised implementation of the same scheme, on one
GPU.** Not that this library is faster than EGGROLL. Nothing here would survive being quoted
as a percentage.

The naive and evosax columns are a memory result rather than a speed one: both fall over at
shapes both low-rank arms handle, on a 16 GB card. That is the `ravel_pytree` argument from
`docs/00` showing up as an allocation failure rather than as an argument.

### What this does not establish

**It is not on the A100 node, so it cannot join the M4 table above.** That table is 8x
A100-SXM4-80GB, where `mirrored_lr1/B` at `d=512, N=1024` reads 10.26 ms against 41.17 here, a
4x hardware gap. Merging the two would be exactly the stitching this document has spent three
sections warning against. **Criterion 3 stays "met in the letter" until M4 is re-run with this
arm on one node**, which is a single-GPU booking rather than an 8x one, since M4's like-for-like
comparison is `--devices 1`.

**D=1 only, so nothing here is about sharding.** The EGGROLL arm has no mesh path in this
harness, which is fair at one device and would not be at eight. The interesting question that
follows, and that this measurement sets up rather than answers, is whether the parity survives
when this library shards and theirs does not.

**Throughput, not solution quality.** The arms do not take the same step: their update z-scores
fitness and applies optax, this one uses centred ranks and plain SGD. `m4.py` has said so since
it was written and it is not fixed by this arm.

**One thing is measured and unexplained.** `cost_analysis` puts the two arms within 0.4% at
`d=512, N=256` and 15% apart at `d=2048, N=256`, where shardes does *fewer* FLOPs for 2% more
wall clock. The likely cause is dull: their update contracts `N` rank-1 outer products per
matrix, roughly 12.9 GFLOP at that shape, which is the order of the gap, and the arms were
never claimed to take the same step. It is recorded rather than claimed, because this document
has already been wrong once about what `cost_analysis` counts.

### Status after the EGGROLL arm

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met**, 0.292 to 0.939 at `D=8`, one commit |
| 2 | crossover measured, phase diagram | **met**, no contour: the grid is too coarse |
| 3 | one comparison against an external reference | **met** by evosax; EGGROLL now measured, but on the wrong hardware to report beside M4 |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | **still needs rewriting, in Andres's words** |

Criterion 3's spirit now needs one single-GPU session rather than an unwritten adapter, which
is the part worth recording: the blocker was a wrong belief about their code, not work.


## The EGGROLL arm on the benchmark hardware, 2026-08-14

The section above ends with "criterion 3 stays met in the letter until M4 is re-run with
this arm on the benchmark node, which is a single-GPU booking". This is that booking:
one A100-SXM4-80GB (driver 595.71.05, CUDA 13.2, community cloud, jax 0.11.0, the same
`XLA_FLAGS` as every prior session), 33 minutes of uptime, $0.76. Two runs, both at
commit `a858998`, every record `dirty_worktree: false`.
`experiments/phase2/results-m4-a100-run1/` and `-run2/`.

ms/gen at `--devices 1`, run 1 / run 2:

| shape | shardes `mirrored_lr1/B` | EGGROLL `rank1` | ratio | naive ES (run 1) | evosax (run 1) |
|---|---|---|---|---|---|
| `d=512, N=256` | 3.12 / 3.08 | **2.96 / 3.01** | 1.052 / 1.022 | 12.59 | 12.25 |
| `d=512, N=1024` | **10.10 / 10.07** | 10.35 / 10.26 | 0.977 / 0.982 | 47.72 | 45.35 |
| `d=2048, N=128` | 11.12 / 11.04 | **10.74 / 10.74** | 1.036 / 1.028 | 80.33 | 87.26 |
| `d=2048, N=256` | 20.48 / 20.49 | 20.48 / 20.44 | 1.000 / 1.003 | 160.11 | 173.83 |

**Parity on the hardware the rest of this document reports.** The ratio spans 0.977 to
1.052 and the lead alternates by shape. Unlike the laptop measurement, the two runs
agree to about 1% everywhere, including the smallest shape, so the spread between the
arms is now larger than the noise and it still does not favour either side. Both
low-rank implementations are 4x faster than the naive and evosax references at `d=512`
and 8x at `d=2048`, from the same factorisation.

**These numbers can sit beside the existing M4 table, and the overlap proves it.** The
prior session measured `mirrored_lr1/B` at `d=512, N=1024` as 10.26 ms on the 8x node;
this session reads 10.10 / 10.07 on one GPU of the same model and driver. evosax: 45.04
there, 45.35 / 46.96 here. Same program, same hardware, agreement within about 2%,
which is what "the D=1 rows do not depend on the other seven GPUs" predicts.

**On an 80 GB card the naive and evosax arms complete at every shape**, so the memory
verdict from the laptop session (both OOM at three of four shapes on 16 GB) and the
throughput verdict now exist separately: they are 4x to 8x slower where they run at
all, and they stop running where the low-rank arms continue.

### Status after the benchmark-node run

| | criterion | status |
|---|---|---|
| 1 | strong and weak curves, efficiency stated | **met**, 0.292 to 0.939 at `D=8`, one commit |
| 2 | crossover measured, phase diagram | **met**, no contour: the grid is too coarse |
| 3 | one comparison against an external reference | **met, in the letter and in the spirit**: EGGROLL's own implementation, benchmark hardware, two runs |
| 4 | reproducible from a committed config plus a recorded environment | **met** |
| 5 | a limitations paragraph a skeptic would accept as fair | **still needs rewriting, in Andres's words** |

Criterion 3 is now the result it was defined to be: within EGGROLL's own throughput
while offering a general API, measured with their unmodified code at matched shapes on
the document's hardware. Criterion 5 remains the only gate item, and it is prose.
