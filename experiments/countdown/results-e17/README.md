# E17: the contraction crossover on the real model. TPU v5e-8, 2026-08-22

31 of 32 cells of `e17.yaml` (predictions frozen in its header at 63fa651
before the run), one free Kaggle TPU v5e-8 session, kernel
`al252130/shardes-e17-tpu`. A complete production update (ask, NLL on E15's
frozen batch, tell) on Qwen2.5-0.5B, matmul precision highest, 7 timed
repeats per cell.

## Both frozen predictions confirmed

log10(t_B / t_A), medians; negative means B (partial contraction +
model-sized all-reduce) wins:

| arm | N | D=1 | D=2 | D=4 | D=8 |
|---|---|---|---|---|---|
| full rank (seed) | 32 | -0.000 | -0.020 | -0.067 | -0.137 |
| full rank (seed) | 240 | -0.000 | -0.027 | -0.078 | -0.164 |
| rank 1 | 32 | OOM | +0.094 | +0.134 | +0.212 |
| rank 1 | 240 | OOM | OOM | OOM | A: OOM, B: killed |

- **Seed-regenerated: B wins at every D >= 2 and the advantage grows
  monotonically with D**, the committed prediction and the same monotone
  shape as the synthetic block's F2b. At D=1 the two placements tie to
  0.02%, as they must (no communication, same arithmetic).
- **Rank 1: A wins at every measured D, and grows with D** (+0.21 at D=8
  against the synthetic block's +0.16 at d=2048). The frozen prediction
  said "A ahead or parity on the TPU"; the answer is ahead, decisively.
  Qwen's heterogeneous shapes did not move the sign.
- **The memory column is a finding, but not the one written here first.**
  Rank 1 on the real model does not fit one 16 GB chip even at N=32 (both
  placements OOM at D=1, recorded), and N=240 fits nowhere below D=8; the
  seed arm runs at every cell. This was read as the storage-for-compute
  trade at Qwen scale. It is not: E17b extended the grid to three ranks
  and they OOM at identical cells with identical temporaries, and
  `../results-e17b-memory/` shows the term is per-member activations, set
  by how many members are evaluated at once. `mirrored_seed` is
  `SeedRegenerated(chunk=1)` and scans one member at a time; `LowRank` has
  no chunk and batches the per-device population, which is where its speed
  comes from. Same reason the D=8, N=32 speedup below is not a like-for-like
  comparison of perturbation schemes.
- One cell is absent rather than recorded: rank 1, B, N=240, D=8. The
  driver was killed without a traceback while compiling it (host-side
  kill); its A sibling exceeded HBM at the same shape, so the cell is
  infeasible on this chip either way. Not rerun; a second session for one
  doomed cell buys nothing.

## The one-line reading

The A/B crossover measured on the simplified block transfers to a real
0.5B transformer with both signs intact: expensive perturbations want the
model-sized all-reduce, cheap structured perturbations want scalar
gathering, and the gap widens with device count in both directions. As a
bonus scale anchor: at D=8, N=32, the fast configuration (rank 1 under A,
71.6 ms) runs 9.4x faster end to end than the seed arm's best placement
(B, 670 ms). Read that as a configuration-to-configuration figure, not as
the price of the perturbation scheme: the seed arm is scanning its members
one at a time and the rank-1 arm is evaluating all of them at once, so the
9.4x contains the chunk setting as well as the perturbation.
