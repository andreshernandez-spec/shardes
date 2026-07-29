"""SeedRegenerated: Qiu et al. Full-rank noise stored as seeds, regenerated on the fly.

Store only the per-member seed; regenerate the noise for both perturbation and update, and
apply perturb/restore layer by layer in place. Storage drops to a few bytes per member.

Members are genuinely different weight matrices, so they cannot share a base GEMM. That is
the price of keeping full-rank noise, and it caps the population where EGGROLL's does not
(N = 30 in the paper).

Member i derives from fold_in(base_key, i) with i the global member index. This strategy is
also what makes contraction Strategy A possible at all.
"""
