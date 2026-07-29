"""Fitness shaping: centered ranks, plus the group-relative (GRPO-style) variant.

Group-relative shaping is the thing that makes ES competitive with GRPO on reasoning, and
it is verified absent from evosax's shaping module. Small surface, high leverage
(docs/02-phase1-sharded-core.md C1.6).

Centered ranks need a global sort over all N fitnesses, so this is a synchronization
barrier: an all_gather of N scalars plus a wait, every generation. Cheap in bytes, not
free in latency. Phase 2 measures what it costs (E10).

Shaping is also discontinuous in epsilon, which is why Phase 0 sweeps with and without it.
QMC's advantage rests on bounded Hardy-Krause variation and rank transforms destroy it
(docs/00-context.md, obstacle 2).
"""
