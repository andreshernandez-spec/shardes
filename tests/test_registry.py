"""The strategy registry, and the sweep-grid constraint it enforces.

`check_entry` is tested against real inputs rather than only being looped over the
registry, which is empty. A loop over an empty dict passes without checking anything, and
a test that cannot fail is not a test (docs/conventions.md).
"""

import pytest

from shardes.strategies.registry import FULL, STRATEGIES, Entry, check_entry


def entry(rank, scheme):
    return Entry(build=lambda: None, rank=rank, scheme=scheme)


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
