# T7 regeneration decomposition: where seed regeneration's time goes

The cost surface (`results-cost/`, `results-cost-tpu-v5e8/`) measured `seed_regenerated`
at 5-6x the materializing `iid_gaussian` baseline per device on the v5e against 1.3-3x
on the A100, and the paper reported the gap without a cause. Both arms apply a full-rank
perturbation per member; the regenerating arm draws its noise twice per generation
(evaluate and contract) instead of reading it from memory once. `regen_decompose.py`
times, at the cost surface's own cells and discipline (`cost.measure`, bfloat16,
default matmul precision, one device), three things per cell:

- `t_iid`, one generation with materialized i.i.d. noise
- `t_seed`, one generation with seed-regenerated noise
- `t_rng`, drawing N x P float32 normals with the library's counter RNG, member by
  member under a scan, each draw reduced so nothing can be sliced away

and reports `(t_seed - t_iid) / (2 t_rng)`: near one, the gap is random-number
throughput; well above one, the regenerating program costs more than its draws; well
below one, regeneration is cheaper inside the program than the isolated draw.

The `t_iid` and `t_seed` columns reproduce the cost surface to within 2% on both
platforms, so the decomposition reads directly against it.

## A100-SXM4-80GB (RunPod community), commit 6718d82, clean, jax 0.11.1

| cell | t_iid | t_seed | seed/iid | t_rng | gap / 2 t_rng |
|---|---|---|---|---|---|
| d=512, N=256 | 11.8 ms | 37.8 ms | 3.2x | 7.8 ms | 1.67 |
| d=512, N=1024 | 39.8 ms | 122.6 ms | 3.1x | 30.0 ms | 1.38 |
| d=2048, N=256 | 152.5 ms | 201.0 ms | 1.3x | 70.1 ms | 0.35 |
| d=2048, N=1024 | OOM | | | | |

On the A100 the draws account for most of the gap at d=512 (the regenerating program
costs 1.4-1.7x its two draws) and more than all of it at d=2048, where the generation
inside the fused program costs a third of the standalone draw: regeneration overlaps
with the matmuls it feeds. The 1.3-3x of the cost surface is, on this platform, the
price of the random numbers and not much else.

## TPU v5e-8 (Kaggle, one chip), commit 1ba0dd0, re-measured 2026-08-31

| cell | t_iid | t_seed | seed/iid | t_rng | gap / 2 t_rng |
|---|---|---|---|---|---|
| d=512, N=256 | 23.5 ms | 137.2 ms | 5.8x | 13.9 ms | 4.07 |
| d=512, N=1024 | 91.8 ms | 546.3 ms | 6.0x | 54.1 ms | 4.20 |
| d=2048, N=256 | 377.1 ms | 1967.2 ms | 5.2x | 195.5 ms | 4.07 |
| d=2048, N=1024 | OOM (36 GiB of HLO temporaries against 16 GiB) | | | | |

**The timer fix changed nothing here, and that is the result.** `sliced-timer/` holds the
same four cells under the first version of the timer, which touched two elements of each
draw instead of reducing it; they agree with the table above to within 1% on every column
(ratios 4.06, 4.17, 4.05 against 4.07, 4.20, 4.07). On the A100 the same change moved
`t_rng` by a factor of two to eleven, because XLA sliced the generation down to the
elements the timer touched. So the slicing was GPU-compiler behaviour, not a flaw in the
measurement everywhere, and the v5e numbers were right the first time. They were re-run
rather than assumed, which is the only way that sentence can be written.

The two sets are kept apart because `regen_decompose.py` skips a cell whose output file
exists: leaving the old records in place made the e17b kernel's second session skip all
four instead of re-measuring them.

The shape differs from the A100: the regenerating arm costs about four times its two
draws at every cell, and the per-device ratio is 5-6x where the A100's is 1.3-3x. The v5e
gap is a property of the program XLA builds around per-member regeneration on the TPU
(the two passes and their traffic, not the random numbers), and it was not profiled
further.
