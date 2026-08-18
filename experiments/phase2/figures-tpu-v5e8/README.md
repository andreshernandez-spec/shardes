# Figures from the TPU v5e-8 sweep

    python plot.py --results results-tpu-v5e8 --out figures-tpu-v5e8

Same script as the canonical A100 figures, different platform. Worth reading side by
side with `figures/`:

- M1: iid_gaussian/B has no D=1 point at d=2048, N=256 (the recorded OOM in
  `results-tpu-v5e8/README.md`), so its efficiency is normalized to its own smallest
  measured device count, D=2. That cell is why the efficiency formula in `plot.py` is
  D0*T_D0/(D*T_D) rather than T_1/(D*T_D).
- M3: the contraction crossover moves with the platform. On the v5e-8, B wins everywhere
  for iid_gaussian and seed_regenerated, but A is ahead for both low-rank strategies at
  d=2048 (log10(tB/tA) up to +0.16). On the 8x A100 sweep B won 10 of 16 cells. The
  crossover being platform-dependent is the reason M3 is a measured surface, not a rule.
