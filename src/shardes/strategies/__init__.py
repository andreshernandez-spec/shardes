"""Perturbation strategies.

Three implementations and one wrapper (docs/01-phase0-estimator-harness.md C0.1):

    IIDGaussian       full rank, materialized       textbook OpenAI-ES
    SeedRegenerated   full rank, transient          Qiu et al.
    LowRank(r)        rank r, never materialized    EGGROLL

    Mirrored(inner)   antithetic pairs, halves effective N

Sample design across members is *not* a wrapper. It is a `shardes.coupling.Coupling` passed
to a strategy's constructor, because it changes the perturbation directions rather than their
signs and so cannot sit on top of an opaque inner perturbation. That file carries the
argument; docs/04 C3.1 records it as a finding.

The two published algorithms should end up two lines of config apart:

    ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, ...)        # Qiu et al.
    ShardedES(strategy=Mirrored(LowRank(r=1)), n=262_144, ...)        # EGGROLL

If switching between them is not close to this, the abstraction failed and gets reworked
before going further.

`protocol.py` is the interface. `registry.py` is the list of everything implemented, which
both the test suite and the Phase 0 sweep iterate over: register a new strategy there or
it goes untested.
"""
