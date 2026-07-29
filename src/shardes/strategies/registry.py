"""The one list of strategies. Add a strategy here the moment it exists.

Everything in tests/test_strategies.py then applies to it automatically, which is the
point: a strategy that is implemented but unregistered is a strategy that is silently
untested, and the property suite is the only thing standing between a plausible-looking
`sample` and a broken seed contract.

This lives in the library rather than in tests/ because it is not test-only. The E1 sweep
in experiments/phase0/ iterates the same set as its `scheme` x `rank` axes
(docs/01-phase0-estimator-harness.md C0.5), and two copies of that list would drift.

Entries are zero-argument factories, not instances, so importing the registry to read its
ids costs nothing and each use gets a fresh object.

Registration is an explicit dict rather than a decorator on each class, because the
interesting entries are compositions. `Mirrored(LowRank(r=1))` is a strategy under test
and there is no class to hang a decorator on.
"""

from typing import Callable

from shardes.strategies.protocol import PerturbationStrategy

StrategyFactory = Callable[[], PerturbationStrategy]

# id -> factory.
#
# The id lands in pytest test ids and in E1 result filenames, so it is part of the record:
# renaming one orphans every result already on disk. Pick it once.
#
# Expected shape once the strategies exist, from docs/01 C0.1 and C0.5:
#
#     "iid_gaussian":            lambda: IIDGaussian(),
#     "seed_regenerated":        lambda: SeedRegenerated(),
#     "lowrank_r1":              lambda: LowRank(r=1),
#     "lowrank_r4":              lambda: LowRank(r=4),
#     "mirrored_lowrank_r1":     lambda: Mirrored(LowRank(r=1)),
#     "mirrored_hd_lowrank_r1":  lambda: Coupled(Mirrored(LowRank(r=1)), "orthogonal_hd"),
#
# When the E1 driver lands it will need `rank` and `scheme` per entry to build the
# non-rectangular grid in C0.5. Add those as fields here rather than rebuilding the grid
# in experiments/phase0/run.py.
STRATEGIES: dict[str, StrategyFactory] = {}
