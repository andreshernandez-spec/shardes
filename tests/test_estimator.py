"""Estimator quality against exact gradients. The statistical half of Phase 0.

    test_quadratic_estimator_unbiased   for each strategy, mean(g_hat) over many seeds
                                        approaches H theta, relative error < 2% at R = 2000
    test_lowrank_converges_to_fullrank  LowRank(r) approaches IIDGaussian as r grows, gap
                                        decreasing, consistent with O(1/r)
    test_mirrored_cancels_odd           on an odd f, mirrored variance is far below i.i.d.

Fast versions only. The full sweep over N x rank x scheme x shaping x sigma is
experiments/phase0/, not here.
"""
