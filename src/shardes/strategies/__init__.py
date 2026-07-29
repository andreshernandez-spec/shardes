"""Perturbation strategies.

Three implementations and two wrappers (docs/01-phase0-estimator-harness.md C0.1):

    IIDGaussian       full rank, materialized       textbook OpenAI-ES
    SeedRegenerated   full rank, transient          Qiu et al.
    LowRank(r)        rank r, never materialized    EGGROLL

    Mirrored(inner)         antithetic pairs, halves effective N
    Coupled(inner, kind)    sample design across members

The two published algorithms should end up two lines of config apart:

    ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, ...)        # Qiu et al.
    ShardedES(strategy=Mirrored(LowRank(r=1)), n=262_144, ...)        # EGGROLL

If switching between them is not close to this, the abstraction failed and gets reworked
before going further.
"""
