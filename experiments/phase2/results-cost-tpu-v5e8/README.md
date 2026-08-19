# E8 on TPU v5e, session 1 of 2 (Kaggle), 2026-08-18

232 of the 240 cells of `cost-sweep-tpu.yaml`, from the `kaggle/t2cost/` kernel pinned
at d908c1a: 183 measured, 49 recorded undersized, budget stop at 6.79 h against the 6 h
internal budget (the check runs between cells and cell 232, seed_regenerated at d=4096
N=16384, ran long across the line). The 8 unvisited cells are the d=4096, N=16384
mirrored arms; a resume session pinned on a commit containing this directory finishes
them. Platform: TPU v5 lite, one chip (cost.py is D=1), jax 0.11.1, clean worktree,
`matmul_precision: default` recorded per cell.

What the surface says against the GPU run (`results-cost/`), same grid, same driver:

- **C4's suspected inversion does not happen.** The low-rank rewrite pays MORE on the
  v5e, not less: lr1 runs at 0.05x to 0.18x of the dense baseline where both fit,
  against 0.13x to 0.42x on the A100. The MXU making dense matmuls cheap does not
  close the gap; the dense baseline's N distinct matrices are as hostile to the MXU
  as to tensor cores, and 16 GB makes it infeasible over most of the grid besides.
- `seed_regenerated` costs 5x to 6x the materializing baseline at D=1 here, against
  1.3x to 3x on the A100: regeneration is comparatively heavier on this platform.
  It remains the only strategy that never goes undersized, at either memory size.
- The feasibility ceiling moves exactly as 16 GB against 80 GB predicts: dense OOMs
  from N=1024 at d=1024, and even mirrored low-rank loses the largest cells.
- **Anomaly, diagnosed:** `mirrored_lr1` is undersized at cells where `mirrored_lr4`
  and `lr16` run (d=512 N=16384, d=2048 N=4096). `probe_lr1.py` reproduces it with no
  TPU (libtpu compiles for a v5e topology from any linux box; the OOM is a
  compile-time static allocation failure, so it reproduces exactly). The perturbation
  is never materialized at any rank (invariant 3 holds); the excess is activations.
  r=2 and r=4 share one graph structure while r=1 alone holds a third more
  activation-sized f32 temporaries, consistent with XLA canonicalizing the r=1
  correction's contracting-dim-1 dot `(x @ B) @ A.T` into a broadcast-multiply chain
  that neither stays bf16 nor fuses on TPU. The GPU compiler keeps the same shapes
  monotonic in r, which is why the A100 surface shows no anomaly. Any fix is a
  strategy-code decision (expressing the r=1 correction to survive canonicalization).

Regenerate/resume: push `kaggle/t2cost/` at a commit containing this directory
(fill the username at push time), `--accelerator TpuV5E8`.
