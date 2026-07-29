"""The strategy registry, and the sweep-grid constraint it enforces.

`check_entry` is tested against real inputs rather than only being looped over the
registry, which is empty. A loop over an empty dict passes without checking anything, and
a test that cannot fail is not a test (docs/conventions.md).
"""

import pytest

from shardes.coupling import Gaussian, OrthogonalHD, ScrambledSobol
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.registry import FULL, REPRESENTATIVES, STRATEGIES, Entry, check_entry


def entry(rank, scheme):
    return Entry(build=lambda: None, rank=rank, scheme=scheme)


def coupling_of(strategy):
    """The noise source at the bottom of a wrapper chain."""
    while not hasattr(strategy, "coupling"):
        strategy = strategy.inner
    return strategy.coupling


def test_rejects_full_rank_sobol():
    """The non-rectangular grid in docs/01 C0.5, enforced rather than remembered.

    Full-rank sampling is in R^(mn) and direction-number tables stop around 21k
    dimensions, so this cell is not merely undesirable, it is not constructible.
    """
    with pytest.raises(ValueError, match="sobol is low-rank only"):
        check_entry("bad", entry(FULL, "mirrored+sobol"))


def test_allows_low_rank_sobol():
    check_entry("ok", entry(1, "mirrored+sobol"))
    check_entry("ok", entry(4, "mirrored+sobol"))


def test_allows_full_rank_orthogonal_hd():
    """orthogonal_hd is O(d log d) and dimension-agnostic, so it carries the G0
    comparison across both panels."""
    check_entry("ok", entry(FULL, "mirrored+orthogonal_hd"))


@pytest.mark.parametrize("scheme", ["iid", "mirrored", "mirrored+orthogonal_hd"])
def test_allows_full_rank_non_sobol_schemes(scheme):
    check_entry("ok", entry(FULL, scheme))


@pytest.mark.parametrize("rank", [0, -1, 1.5, "r1", None])
def test_rejects_nonsense_ranks(rank):
    with pytest.raises(ValueError, match="rank must be"):
        check_entry("bad", entry(rank, "iid"))


def test_every_registered_entry_is_valid():
    """Vacuous while the registry is empty, which is why check_entry is tested directly
    above. This is the guard that fires when entries start landing."""
    for name, e in STRATEGIES.items():
        check_entry(name, e)


def test_entry_is_frozen():
    """Ids end up in result filenames; a registry someone can mutate at runtime makes
    the record unreliable."""
    e = entry(1, "iid")
    with pytest.raises(Exception):
        e.rank = 2


# --------------------------------------------------------------------------------------
# The grid, and the labels that describe it.
# --------------------------------------------------------------------------------------

# docs/01 C0.5, transcribed. Written out rather than generated from a product, because it
# is not a product: that is the whole point of it being non-rectangular.
GRID = {
    (FULL, "iid"),
    (FULL, "mirrored"),
    (FULL, "mirrored+orthogonal_hd"),
    (4, "iid"),
    (4, "mirrored"),
    (4, "mirrored+orthogonal_hd"),
    (4, "mirrored+sobol"),
    (1, "iid"),
    (1, "mirrored"),
    (1, "mirrored+orthogonal_hd"),
    (1, "mirrored+sobol"),
}

def test_registered_cells_are_exactly_the_documented_grid():
    """Exact in both directions: a cell going missing fails, and so does a cell appearing that
    the doc does not have. The E1 driver iterates this registry, so a missing cell is a missing
    curve in F5 and a spurious one is a config that was never committed."""
    got = {(e.rank, e.scheme) for e in STRATEGIES.values()}
    assert got == GRID


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_scheme_label_matches_the_strategy_it_builds(name):
    """The drift this catches is silent and would ruin the figure: an entry labelled
    `mirrored+orthogonal_hd` whose builder forgot the coupling plots a duplicate of the
    `mirrored` curve, and two curves lying on top of each other reads as a negative result.
    """
    e = STRATEGIES[name]
    s = e.build()
    assert isinstance(s, Mirrored) == ("mirrored" in e.scheme), f"{name}: mirrored label"

    coupling = coupling_of(s)
    assert isinstance(coupling, OrthogonalHD) == ("orthogonal_hd" in e.scheme), f"{name}: hd label"
    assert isinstance(coupling, ScrambledSobol) == ("sobol" in e.scheme), f"{name}: sobol label"
    if "+" not in e.scheme:
        assert isinstance(coupling, Gaussian), f"{name}: unlabelled coupling"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_rank_label_matches_the_strategy_it_builds(name):
    """`rank` is the x-axis of F5 through `d_eff`, so a wrong label moves a point rather
    than failing."""
    e = STRATEGIES[name]
    s = e.build()
    inner = getattr(s, "inner", s)
    if e.rank == FULL:
        assert not isinstance(inner, LowRank), f"{name}: labelled full rank but is LowRank"
    else:
        assert isinstance(inner, LowRank) and inner.r == e.rank, f"{name}: rank mismatch"


def test_representatives_are_registered():
    assert REPRESENTATIVES <= set(STRATEGIES)
