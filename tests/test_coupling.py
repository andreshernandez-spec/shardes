"""Coupled sampling. Phase 0 scope is unsharded; the sharded tests arrive with Phase 3.

    test_coupled_reduces_to_iid     with coupling disabled, bitwise identical to the
                                    uncoupled strategy
    test_coupled_unbiased           E[g_hat] approaches grad f for every kind. Scrambling
                                    is what makes this true for Sobol; deterministic Sobol
                                    must fail it
    test_sobol_first_two_moments    scrambled Sobol through the inverse normal CDF has the
                                    right mean and covariance, and 2-D projections are
                                    equidistributed

Phase 3 adds (docs/04-phase3-coupling.md): test_coupled_device_invariant,
test_hd_block_orthogonality, test_sobol_skip_ahead, test_wrapper_does_not_touch_core.
Do not write those until Gate G0 says yes.
"""
