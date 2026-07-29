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

from shardes.coupling import OrthogonalHD, ScrambledSobol
from shardes.dimensions import FULL
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.protocol import PerturbationStrategy
from shardes.strategies.seed_regenerated import SeedRegenerated

StrategyFactory = Callable[[], PerturbationStrategy]


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
# This is the whole non-rectangular grid from docs/01 C0.5, minus the two sobol cells which
# land with `ScrambledSobol`. `orthogonal_hd` appears only composed with `mirrored`, which is
# the grid's choice and not an oversight: mirroring is the honest baseline, so coupling is
# measured as an increment on top of it rather than against an unmirrored strawman.
#
# Coupling arrives as a constructor argument rather than a `Coupled(...)` wrapper. It changes
# the perturbation *directions*, so it cannot be layered over an opaque inner perturbation
# the way `Mirrored`'s sign flip can. See src/shardes/coupling.py and docs/04 C3.1.
STRATEGIES: dict[str, Entry] = {
    # The class itself is the zero-argument factory.
    "iid_gaussian": Entry(build=IIDGaussian, rank=FULL, scheme="iid"),
    "seed_regenerated": Entry(build=SeedRegenerated, rank=FULL, scheme="iid"),
    "lowrank_r1": Entry(build=lambda: LowRank(r=1), rank=1, scheme="iid"),
    "lowrank_r4": Entry(build=lambda: LowRank(r=4), rank=4, scheme="iid"),
    "mirrored_full": Entry(build=lambda: Mirrored(IIDGaussian()), rank=FULL,
                           scheme="mirrored"),
    "mirrored_lr1": Entry(build=lambda: Mirrored(LowRank(r=1)), rank=1,
                          scheme="mirrored"),
    "mirrored_lr4": Entry(build=lambda: Mirrored(LowRank(r=4)), rank=4,
                          scheme="mirrored"),
    "mirrored_hd_full": Entry(build=lambda: Mirrored(IIDGaussian(coupling=OrthogonalHD())),
                              rank=FULL, scheme="mirrored+orthogonal_hd"),
    "mirrored_hd_lr1": Entry(build=lambda: Mirrored(LowRank(r=1, coupling=OrthogonalHD())),
                             rank=1, scheme="mirrored+orthogonal_hd"),
    "mirrored_hd_lr4": Entry(build=lambda: Mirrored(LowRank(r=4, coupling=OrthogonalHD())),
                             rank=4, scheme="mirrored+orthogonal_hd"),
    "mirrored_sobol_lr1": Entry(build=lambda: Mirrored(LowRank(r=1, coupling=ScrambledSobol())),
                                rank=1, scheme="mirrored+sobol"),
    "mirrored_sobol_lr4": Entry(build=lambda: Mirrored(LowRank(r=4, coupling=ScrambledSobol())),
                                rank=4, scheme="mirrored+sobol"),
}


# The fast test tier runs the shared property suite over these only; the rest are marked
# slow and run in the full suite. Each representative covers a distinct implementation
# path: iid_gaussian materializes, seed_regenerated scans, mirrored_lr1 is a wrapper over
# the factored path and so exercises LowRank transitively, mirrored_hd_lr1 adds the coupled
# noise source and LowRank's per-column design families. The rest are recombinations of paths
# already covered, and test_lowrank.py and test_coupling.py cover LowRank and OrthogonalHD
# directly regardless.
#
# This is a test-speed policy, but it is knowledge about which strategies are redundant
# with which, so it lives with the strategies rather than in a test file.
REPRESENTATIVES = frozenset(
    {"iid_gaussian", "seed_regenerated", "mirrored_lr1", "mirrored_hd_lr1"}
)
