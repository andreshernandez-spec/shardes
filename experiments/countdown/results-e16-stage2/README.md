# E16 stage 2: the correction is portable to first order. 1.5B, 2026-08-22

Four cells of `e16-stage2.yaml` (predictions frozen in its header at 886ff81
before the run), one community A100-SXM4-80GB, ~12 h (~$17), pod deleted.
Qwen2.5-1.5B (port validated by the golden-logit test before launch), E15's
original frozen batch, member_chunk 10.

## Against the frozen predictions

| arm | N | measured (median [range]) | raw fit | ratio | calibrated (x0.53) | ratio |
|---|---|---|---|---|---|---|
| full rank | 30 | 8.3e-5 [7.5-8.5] | 8.7e-5 | 0.95x | -- | -- |
| full rank | 240 | 2.3e-4 [2.0-2.6] | 2.5e-4 | 0.92x | -- | -- |
| rank 1 | 30 | 7.7e-5 [3.6-11] | 2.0e-4 | 0.39x | 1.0e-4 | 0.74x |
| rank 1 | 240 | 2.3e-4 [1.9-2.4] | 5.5e-4 | 0.41x | 2.9e-4 | 0.78x |

The three frozen checks, in their committed order:

1. **The N/d_eff law transfers unrefitted.** Within-arm N=240/N=30 is 2.80x
   (full rank) and 2.94x (rank 1) against sqrt(8) = 2.83. At 3.1x the
   parameters of the model the curves were bridged on, the slope holds.
2. **The calibrated prediction lands where the raw one misses.** Raw
   low-rank predictions are off by 2.4-2.6x at 1.5B; the 0.5B-calibrated
   predictions are within 22-26%. The correction is portable to first
   order, with a mild scale drift: the effective correction is ~0.40 at
   1.5B against 0.53 at 0.5B. One constant carries most of the transfer;
   it is not perfectly scale-invariant, and saying so is the result.
3. **Full rank stays on its raw prediction** (0.92-0.95x), extending E15's
   0.91-0.94x from 0.5B to 1.5B with no correction at all.

A fourth, unasked-for regularity: rank parity holds on the real model at
1.5B too. At matched N the rank-1 and full-rank cosines are within each
other's ranges (7.7 vs 8.3e-5; 2.27 vs 2.32e-4), the same tie E13/E15
measured at 0.5B, now one model size up.

## Reading

The estimator geometry the paper argues from is not an artifact of one
model: the isotropic (full-rank) fit predicts a 1.5B transformer to within
8%, the scaling law needs no refit, and the one systematic unknown, the
low-rank projection's extra cost on real loss surfaces, moves slowly enough
with scale that a single 0.5B calibration lands within a quarter. A
two-point (0.5B, 1.5B) fit of that correction against d_model is the
obvious next step if a third model size ever matters.
