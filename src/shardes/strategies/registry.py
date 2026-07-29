"""The one list of strategies. Add a strategy here the moment it exists.

Everything in tests/test_strategies.py then applies to it automatically, which is the
point: a strategy that is implemented but unregistered is a strategy that is silently
untested, and the property suite is the only thing standing between a plausible-looking
`sample` and a broken seed contract.

This lives in the library rather than in tests/ because it is not test-only. The E1 sweep
in experiments/phase0/ iterates the same set as its `scheme` x `rank` axes
(docs/01-phase0-estimator-harness.md C0.5), and two copies of that list would drift.

`build` is a zero-argument factory, not an instance, so importing the registry to read its
ids constructs nothing. Registration is an explicit dict rather than a decorator per
class, because the entries that matter are compositions: `Mirrored(LowRank(r=1))` is a
strategy under test and there is no class to hang a decorator on.
"""

from dataclasses import dataclass
from typing import Callable

from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.protocol import PerturbationStrategy

StrategyFactory = Callable[[], PerturbationStrategy]

FULL = "full"


@dataclass(frozen=True)
class Entry:
    """One parametrization of the property suite, and one cell of the E1 sweep.

    rank:   an int r, or FULL.
    scheme: "iid", "mirrored", "mirrored+orthogonal_hd", "mirrored+sobol".

    The sweep grid in docs/01 C0.5 is non-rectangular, and it is expressed here by
    absence: there is simply no entry with rank=FULL and a sobol scheme. `check_entry`
    is what stops one being added by accident.
    """

    build: StrategyFactory
    rank: int | str
    scheme: str


def check_entry(name: str, entry: Entry) -> None:
    """Raise if an entry violates the sweep grid. See docs/01 C0.5."""
    if entry.rank == FULL and "sobol" in entry.scheme:
        raise ValueError(
            f"{name}: sobol is low-rank only. Full-rank sampling is in R^(mn) and every "
            "published direction-number table stops around 21k dimensions, so the scheme "
            "is not constructible there. docs/01-phase0-estimator-harness.md C0.5."
        )
    if entry.rank != FULL and (not isinstance(entry.rank, int) or entry.rank < 1):
        raise ValueError(f"{name}: rank must be a positive int or {FULL!r}, got {entry.rank!r}")


# id -> Entry.
#
# The id lands in pytest test ids and in E1 result filenames, so it is part of the record:
# renaming one orphans every result already on disk. Pick it once.
#
# Expected shape once the strategies exist, from docs/01 C0.1 and C0.5:
#
# Still to come, from docs/01 C0.1 and C0.5:
#
#     "seed_regenerated":   Entry(SeedRegenerated, FULL, "iid"),
#     "mirrored_full":      Entry(lambda: Mirrored(IIDGaussian()), FULL, "mirrored"),
#     "lowrank_r1":         Entry(lambda: LowRank(r=1), 1, "iid"),
#     "mirrored_lr1":       Entry(lambda: Mirrored(LowRank(r=1)), 1, "mirrored"),
#     "mirrored_hd_lr1":    Entry(lambda: Coupled(Mirrored(LowRank(r=1)), "orthogonal_hd"),
#                                 1, "mirrored+orthogonal_hd"),
#     "mirrored_sobol_lr1": Entry(lambda: Coupled(Mirrored(LowRank(r=1)), "sobol_scrambled"),
#                                 1, "mirrored+sobol"),
STRATEGIES: dict[str, Entry] = {
    # The class itself is the zero-argument factory.
    "iid_gaussian": Entry(build=IIDGaussian, rank=FULL, scheme="iid"),
}
