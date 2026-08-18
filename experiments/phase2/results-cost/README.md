# E8: single-device cost surface on A100, 2026-08-18

The full 240-cell grid of `cost-sweep.yaml` (d_model x population x strategy x compute
dtype at D=1), run by `cost.py --allow-partial` on a rented A100-SXM4-80GB
(RunPod community, single GPU), code at c93cf5d, clean worktree. ~2.5 h of pod time.

181 cells measured, 59 recorded undersized (RESOURCE_EXHAUSTED at that shape). The
undersized cells are surface endpoints, not gaps: the sweep is designed to run past the
memory ceiling so the ceiling's position is data. Where it sits confirms the storage
story: `iid_gaussian` loses 23 of its 40 cells and starts losing them at d=512, while
`seed_regenerated` and `lowrank_r1/4/16` (unmirrored) lose none, and the mirrored
low-rank arms give way only at large d x large N where the ceiling is the model and
optimizer replicas, not the perturbations.

Timed cells: 3 warmup, 5 repeats, median and IQR recorded per cell,
`matmul_precision: default` (this surface is about relative cost at the precision a
user would actually run, unlike the scaling sweep which pins highest; the precision is
recorded in every record either way).

Regenerate:

    python cost.py --config cost-sweep.yaml --allow-partial

`--allow-partial` accepts undersized cells as records; without it the driver exits 3
on any incomplete directory.
