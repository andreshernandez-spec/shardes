# E17b: the real-model contraction crossover as a figure. TPU v5e-8, 2026-08-25 to 08-31

All 128 cells of `e17b.yaml` (predictions frozen in its header before any run), five
free Kaggle TPU v5e-8 sessions, kernel `al252130/shardes-e17b-tpu`. One complete
production update per cell (ask, teacher-forced NLL on E15's frozen batch, tell) on
Qwen2.5-0.5B, matmul precision highest, 3 warm-ups discarded and 5 timed repeats.
`plot_e17.py` draws F9. 68 cells timed, 60 recorded OOM.

E17 (`../results-e17/`) measured 31 cells at two populations and two arms; this is the
same driver over four populations and four arms, which turns the appendix table into a
main-text figure.

## log10(t_B / t_A), medians. Negative means B wins

| arm | N | D=1 | D=2 | D=4 | D=8 |
|---|---|---|---|---|---|
| full rank (seed) | 32 | -0.000 | -0.020 | -0.067 | -0.135 |
| full rank (seed) | 64 | -0.000 | -0.024 | -0.073 | -0.151 |
| full rank (seed) | 128 | -0.000 | -0.026 | -0.077 | -0.160 |
| full rank (seed) | 240 | -0.000 | -0.027 | -0.079 | -0.164 |
| rank 1 | 32 | OOM | +0.095 | +0.137 | +0.225 |
| rank 1 | 64 | OOM | OOM | +0.074 | +0.116 |
| rank 1 | 128 | OOM | OOM | OOM | +0.066 |
| rank 1 | 240 | OOM | OOM | OOM | OOM |
| rank 4 | 32 | OOM | +0.094 | +0.136 | +0.212 |
| rank 4 | 64 | OOM | OOM | +0.073 | +0.146 |
| rank 4 | 128 | OOM | OOM | OOM | +0.061 |
| rank 4 | 240 | OOM | OOM | OOM | OOM |
| rank 16 | 32 | OOM | +0.085 | +0.117 | +0.177 |
| rank 16 | 64 | OOM | OOM | +0.040 | +0.077 |
| rank 16 | 128 | OOM | OOM | OOM | +0.004 |
| rank 16 | 240 | OOM | OOM | OOM | OOM |

## The three frozen predictions

**Seed-regenerated: B wins at every D >= 2, growing with D at every N.** Confirmed at
all 16 cells. The new populations 64 and 128 land between N=32 and N=240 as predicted,
and the ordering in N is monotone at every device count. D=1 ties to within 0.05%, as
it must: no communication, same arithmetic.

**The low-rank arms: A wins, growing with D.** Confirmed at all 18 shapes that fit,
six per rank, across four populations. No sign flip anywhere.

**The gap shrinks as rank grows.** Confirmed. The config froze this as the falsifiable
one ("a gap LARGER than rank 1's would" contradict the mechanism), and rank 16 is below
rank 1 at every population it shares: +0.177 against +0.225 at N=32, +0.077 against
+0.116 at N=64, +0.004 against +0.066 at N=128. At N=128, D=8 rank 16 is a tie to 1%,
the closest the grid comes to a crossover, and it approaches it from A's side without
crossing. One non-monotonicity: rank 4 at N=64, D=8 is +0.146, above rank 1's +0.116.
It is the only cell where the ordering in r inverts, and rank 16 is well below both.

A pattern E17 could not see, having only two populations: **the low-rank gap also
shrinks with N**, from +0.225 at N=32 to +0.066 at N=128 (rank 1, D=8). The seed arm's
advantage moves the other way, growing with N. Both are the cost model's doing: A's
saving is per-candidate contraction work and B's cost is a model-sized all-reduce that
does not grow with the population, so raising N feeds the term that favours B in both
arms.

## The memory column, and what it is not

Every OOM is a low-rank arm; the seed arm fits all 16 of its cells. The boundary is a
pure function of members per device and nothing else: **every low-rank cell at 30 or
more members per device is over HBM, every one at 16 or fewer fits**, identically for
r=1, 4 and 16 and for both placements. All 60 OOM records are consistent with it,
including the three measured last.

E17 read this column as the storage-for-compute trade, and `e17b.yaml` froze the
prediction that "rank 16 is expected to OOM where rank 1 runs at the largest N". **That
prediction is wrong, and this is the run's one clear miss.** Rank changes nothing: the
three ranks OOM at the same 10 (N, D) shapes, both placements, with XLA's reported
temporaries agreeing to 0.1%
(114.58G at r=16, 114.61G at r=1, 114.66G at r=4 for N=128, D=1). Memory that does not
move with r is not the factors.

`../results-e17b-memory/` prices what it actually is. The term scales with the prompt
batch, so it is per-member activations, dominated by the `(members, prompts, T-1,
151936)` logits and the f32 log-softmax over them. What sets how many are live is the
evaluation: `mirrored_seed` is built as `SeedRegenerated(chunk=1)` and scans one member
at a time, while `LowRank` has no chunk and evaluates the whole per-device population in
one vmap, which is exactly where its speed comes from, since the base weight is
unbatched under vmap so members share one GEMM. Giving the seed arm the low-rank arms'
batching reproduces their memory curve.

So the trade is real but runs opposite to how E17 told it. The arm with the cheap
perturbation is the one that runs out of memory, because it buys speed by batching
members and batching members costs activations. The factors themselves are far too
small to appear.

This does not touch the ratios above: A and B are the same arm at the same shape, so the
chunking is common to both and cancels. It does reach cross-arm comparisons. At D=8,
N=32 the best low-rank configuration is 71.0 ms against the seed arm's 672.6 ms, and
that 9.5x is between two evaluation strategies as well as two perturbation schemes.
This grid ran the seed arm at chunk 1, the setting that batches least. On E13's decode
workload the chunk is worth 2.26x across its range, more than the perturbation scheme
is worth at matched evaluation (`../probes/results-a100-chunk/`), so 9.5x is inflated
in a known direction. Not by a known amount: that probe decodes 96 tokens where this
scores a teacher-forced NLL, and the two reward batching differently.

## Best-placement wall clock at D=8, ms

| arm | N=32 | N=64 | N=128 | N=240 |
|---|---|---|---|---|
| full rank (seed) | 672.6 | 1276.0 | 2481.7 | 4590.5 |
| rank 1 | 71.0 | 121.9 | 229.9 | OOM |
| rank 4 | 72.4 | 119.3 | 227.4 | OOM |
| rank 16 | 77.8 | 135.8 | 271.9 | OOM |

Ranks 1 and 4 are within 2% of each other at every population; rank 16 costs 10-18%
more. The seed arm is the only one that reaches N=240, at 4.6 s per update.

## Sessions

Five, because a Kaggle TPU session is capped well below what 128 cells of this size
need. Per-cell result files plus skip-on-exists make a session resumable, and the
driver takes a `--budget` so it stops cleanly between cells and exits 2; the kernel
runs it in repeated slices and stops when the grid is done or a slice adds nothing.

| session | SHA | cells added | total |
|---|---|---|---|
| 1 | 1ba0dd0 | 10 | 10 |
| 2 | e2d275d | 30 | 40 |
| 3 | 41b04b9 | 53 | 93 |
| 4 | e6f2c07 | 25 | 118 |
| 5 | 83e602e | 10 | 128 |

Session 2 was killed mid-cell with no traceback after a run of recorded OOMs, host
memory rather than HBM. Session 4 spent all four of its slices and stopped with three
hours of the session unused, which is why the kernel now takes six. Sessions 1 and 2
also ran the phase-2 regeneration decomposition and contraction isolation as a prelude;
those results live in `../../phase2/`.

Cells carry the SHA they were measured at, and two library files moved across the five.
Both are inert here, checked rather than assumed:

- `lowrank.py` gained `PAD_RANK1` at 030732d, which makes the r=1 pad TPU-only where it
  had been unconditional. Every cell here ran on a v5e, so r=1 is padded on both sides
  of that change, and the branch is guarded by `a.shape[-1] == 1`, so ranks 4 and 16
  never reach it at all.
- `make_mesh` gained a multi-host branch for E18. It triggers only when the chosen
  devices span more than one process; a single v5e-8 is one process, so every cell here
  takes the original path.

The remaining diffs are the kernel's slice count and the committed cells the later
sessions skip over.
