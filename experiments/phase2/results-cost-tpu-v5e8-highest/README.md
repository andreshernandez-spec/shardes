# E11 precision ablation: matmul `highest` vs `default`, TPU v5e, 2026-08-19

The 36 cells of `cost-precision-tpu.yaml` (d in {512, 2048} x N in {64, 256, 1024} x
6 strategies, bf16), one T4 session pinned at 72f6ada, one chip. 35 measured, 1
undersized (iid_gaussian at d=2048, N=1024 needs more memory at `highest` than the
default-precision run did, which is itself a data point). Compare cell-for-cell
against `results-cost-tpu-v5e8` at the same (d, N, strategy, bfloat16); `tb3.py`
computes the ratios.

What it says: `highest` is nearly free for the strategies whose time is the base
model GEMM (seed_regenerated 1.02x, mirrored_seed 1.02x, iid_gaussian 1.08x) and
costs 1.6x to 1.9x for the low-rank family (mirrored_lr1 1.85x, lr4 1.76x, lr16
1.64x), whose per-member correction GEMMs are exactly the multiplies the MXU was
doing in bf16. So the scaling sweep's choice to time at `highest` was conservative
for the low-rank arms specifically, and a practitioner running `default` gets the
low-rank speedups in F4, not the flatter `highest` ones.
