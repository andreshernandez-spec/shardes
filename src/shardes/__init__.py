"""shardes: sharded evolution strategies for JAX.

Phases 0 to 2 are implemented: the estimator harness, the sharded ask/eval/tell core, and
the benchmarks behind `docs/03`. Read PLAN.md for the gates, then
docs/02-phase1-sharded-core.md for the API this package exposes.

This said "nothing is implemented yet" until 2026-08-11, which is what a status line in an
executable module does if it is not the thing anyone edits when the status changes.

Every module here carries a docstring saying what belongs in it and which doc section
specifies it. The code is written by hand, phase by phase, against the gates in PLAN.md.
"""
