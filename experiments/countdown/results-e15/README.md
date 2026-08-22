# E15: estimator accuracy on the real model. A100, 2026-08-21/22

Ten cells of `e15.yaml` (5 strategies x N in {30, 240}, 5 replicates each), one
community A100-SXM4-80GB pod at $1.39/h, ~14 h of pod time (~$19) including two
failed attempts. `analysis_e15.py` reproduces every number below from this
directory plus `experiments/phase0/results`; `--latex` writes the paper's
Table 4.

## Session

Three attempts, one pod. Attempts 1 and 2 ran at 5227277 and both died at the
`mirrored_lr1 N=240` cell: the low-rank arms evaluate all members through the
forward at once and the logits for 240 members are ~70 GiB, over the A100's
80 GB with the model and gradient resident. The fix (PR #81, 5cb1b64) evaluates
fitness in even chunks of 30 members, each regenerated from
`(base_key, member_ids)` by the library's own sample/apply, the same
re-derivation `tell`'s contraction uses; per-member fitness does not depend on
batching, so the cosines are unchanged. Attempt 3 resumed at 5cb1b64, skipped
the three finished cells, and completed the remaining seven. Cells therefore
stamp two commits: `mirrored_seed` (both N) and `mirrored_lr1 N=30` at 5227277,
the rest at 5cb1b64. One objective throughout: teacher-forced NLL over the same
fixed 8-prompt batch, sigma 1e-3, one production ask/evaluate/tell step against
`jax.grad` of the same function.

## The bridge, measured (like-for-like slice: centered ranks, sigma 1e-3)

| arm | N=30 measured (x1e-4) | pred | N=240 measured | pred |
|---|---|---|---|---|
| full rank | 1.4 [1.3-1.9] | 1.6 | 4.2 [3.8-4.4] | 4.5 |
| rank 1 | 1.4 [1.2-1.7] | 2.7 | 3.9 [3.8-4.2] | 7.6 |
| rank 4 | 1.9 [0.7-2.0] | 2.8 | 4.4 [4.1-4.8] | 7.9 |
| rank 16 | 1.5 [1.1-1.7] | no curve | 4.1 [3.9-4.5] | no curve |
| rank 1, unpaired | 1.6 [1.1-1.9] | 3.8 | 5.3 [5.1-5.5] | 10.6 |

Four findings:

1. **Full rank transfers within 10%** (ratios 0.91x and 0.94x) across a d_eff
   of 494M, one to three decades below the fitted range. The extrapolation the
   paper flagged as far holds almost exactly where the noise is isotropic.
2. **The low-rank family lands uniformly at ~0.5x the fit** (0.43-0.67x across
   ranks and populations). Same order, one systematic factor: the real
   transformer's loss surface charges roughly 2x more for the low-rank
   projection than the synthetic block did. Because the shortfall is uniform,
   every relative claim transfers: the cross-rank tie at N=30 (all medians
   1.4-1.9e-4, overlapping ranges), the slope (within-arm N-scaling 2.3-3.3x
   against sqrt(8)=2.8), and the ordering.
3. **The unpaired-vs-mirrored ratio survives contact with the real model**:
   5.3/3.9 = 1.36x at N=240 against the phase-0 prediction of 1.34x at matched
   population. The arm EGGROLL's sampler cannot express estimates better per
   sample on Qwen itself.
4. **Rank 16 behaves like its family** (1.5e-4 / 4.1e-4), consistent with
   riding the rank-1/rank-4 trend, as C6b assumed without a curve.
