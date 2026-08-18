# T1: the Phase 2 sweep on TPU v5e-8 (Kaggle), 2026-08-18

The full 256-cell grid of `sweep-tpu.yaml` (identical to `sweep.yaml`, own results dir),
run in a single Kaggle session by the committed kernel
`experiments/phase2/kaggle/t1sweep/`, pinned at c93cf5d. One session covered the whole
grid, so the resume path the kernel was built for went unused.

Platform: 8x TPU v5 lite (v5e-8), jax 0.11.1 (upgraded from the image's 0.10.2 in the
kernel, device count asserted in a fresh interpreter afterwards), clean worktree. Each
JSON carries the full env stamp. Timed generations ran at `matmul_precision: highest`,
same as the A100 sweep, so the two platforms measure the same computation; the
trajectory guard in `run.py` runs at highest regardless.

255 cells measured, 1 recorded error:

- `mode=strong__D=1__d=2048__N=256__s=iid_gaussian__how=B` failed with
  RESOURCE_EXHAUSTED: HLO temporaries need 16.69 GB and a v5e chip has 15.75 GB HBM.
  That is the memory ceiling `sweep-tpu.yaml` predicts for the materializing baseline
  (a v5e chip has a fifth of an A100's HBM, so the ceiling moves down the grid), and it
  lands on the one strategy that has to hold the population. `seed_regenerated`,
  `mirrored_lr1` and `lowrank_r1` all ran at the same point. The record is data, not a
  gap: M6 asks where storage becomes binding, and on this platform this is where.

The driver exited non-zero because of that cell (its contract is "results directory
incomplete"), which Kaggle surfaces as kernel status ERROR. The log confirms all 256
cells were visited: `256 written, 1 failed, 0 over cap, 0 needed more devices`.

Total measured wall time inside cells: 135 min. Session wall time was under the 8 h
budget by a wide margin.

Regenerate: push the kernel in `kaggle/t1sweep/` (fill the username at push time) at a
commit containing this directory's parent config, `--accelerator TpuV5E8`.
