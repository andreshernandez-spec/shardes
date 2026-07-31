"""Couplings as noise sources: `Gaussian`, `OrthogonalHD`, `ScrambledSobol`.

The registry entries carry the coupled strategies through the whole property suite in
test_strategies.py and the unbiasedness suite in test_estimator.py, so nothing here repeats
the seed contract, chunk additivity or linearity. What is here is specific to the point-set
design: that it is orthogonal where it claims to be, only concentrated where it claims to
be, and that turning it off is a no-op.

Phase 0 scope is unsharded. The sharded tests arrive with Phase 3
(docs/04-phase3-coupling.md) and are conditional on Gate G0.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shardes.coupling import (
    GAUSSIAN,
    _SOBOL_BITS,
    Coupling,
    Gaussian,
    OrthogonalHD,
    ScrambledSobol,
    _direction_numbers,
    hadamard_row,
    sobol_point,
)
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.seed_regenerated import SeedRegenerated
from shardes.transforms.fwht import fwht

STREAM = jax.random.key(0)


def rows(coupling: Coupling, ids, d: int, stream: jax.Array = STREAM) -> jax.Array:
    """(len(ids), d) draws, one row per member id.

    `jit` is not an optimisation of the thing under test, it is what makes this file cheap
    enough to keep in the fast tier. A coupling is ~100 primitives (an FWHT chain, or 30 XOR
    steps), and eagerly dispatching each one under vmap compiles a tiny HLO module per
    primitive: 1.2 s for a d=8 case that computes nothing. One compile instead of a hundred
    took the file from 38 s to a few seconds.
    """
    f = jax.jit(jax.vmap(lambda i: coupling(stream, i, d, jnp.float32)))
    return f(jnp.asarray(ids))


def sampled(strategy, base_key, params, member_ids):
    """`strategy.sample`, jitted. Same reason as `rows`: eager dispatch of a few hundred
    primitives costs seconds on arrays this small."""
    return jax.jit(strategy.sample)(base_key, params, member_ids)


# --------------------------------------------------------------------------------------
# hadamard_row: the O(n) shortcut into the HD chain.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 8, 64, 512])
def test_hadamard_row_matches_fwht_of_a_one_hot(n):
    """The identity that lets the chain skip its first butterfly.

    fwht already has an exact oracle against jax.scipy.linalg.hadamard, so checking against
    fwht transitively checks against the dense matrix.
    """
    for p in range(min(n, 8)):
        want = fwht((jnp.arange(n) == p).astype(jnp.float32))
        assert jnp.array_equal(hadamard_row(jnp.int32(p), n), want), f"n={n} p={p}"


def test_hadamard_row_is_plus_minus_one():
    assert jnp.all(jnp.abs(hadamard_row(jnp.int32(5), 32)) == 1.0)


# --------------------------------------------------------------------------------------
# OrthogonalHD: the design property.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [8, 64, 256])
def test_hd_block_is_exactly_orthogonal(d):
    """A whole block's directions are orthonormal to float precision, not approximately.

    docs/04 C3.1 described this as orthogonal "within tolerance" with an O(1/sqrt(d)) band.
    That is wrong and the correction matters: H/sqrt(d) is orthogonal and symmetric, every
    Rademacher D is orthogonal, so the product is *exactly* orthogonal. The O(1/sqrt(d))
    band belongs to cross-block pairs and to how close a row is to Haar, neither of which is
    this. Asserting the weak version would have passed on a broken chain.

    Rows have second moment 1 per entry, so norm^2 is exactly d and the Gram is d*I.
    """
    m = rows(OrthogonalHD(), jnp.arange(d), d)
    gram = (m @ m.T) / d
    assert float(jnp.max(jnp.abs(gram - jnp.eye(d)))) < 1e-4


@pytest.mark.parametrize("d", [16, 128])
def test_hd_row_norm_is_deterministic(d):
    """|eps|^2 == d exactly, where a Gaussian draw gives d +/- sqrt(2d).

    Worth stating separately from orthogonality: it means every member's fitness is
    evaluated at the same radius, so none of the population is spent on an unusually short
    or long step. Under mirrored iid sampling that radius spread is the one source of
    variance mirroring does not remove.
    """
    m = rows(OrthogonalHD(), jnp.arange(d), d)
    norms = jnp.sum(jnp.square(m), axis=1)
    assert float(jnp.max(jnp.abs(norms - d))) < 1e-3 * d


@pytest.mark.slow
def test_hd_blocks_are_independent_and_only_near_orthogonal():
    """Cross-block cosines sit in the O(1/sqrt(d)) band, not at zero.

    The pair of assertions is the point. The upper bound says blocks do not interfere, which
    is what makes a per-device block coupled with no communication (docs/04 C3.2). The lower
    bound says they are genuinely independent draws rather than the same block twice, which
    is the bug a one-sided test would miss: reusing block 0 everywhere passes any
    "small cosine" check with cosine exactly zero.
    """
    d = 256
    a = rows(OrthogonalHD(), jnp.arange(d), d)
    b = rows(OrthogonalHD(), jnp.arange(d, 2 * d), d)
    cos = jnp.abs(a @ b.T) / d
    band = 1.0 / jnp.sqrt(d)

    assert float(jnp.mean(cos)) < 2 * band, "cross-block cosines above the concentration band"
    assert float(jnp.max(cos)) < 8 * band, "cross-block cosines far above the band"
    assert float(jnp.mean(cos)) > 0.2 * band, "suspiciously orthogonal: same block reused?"


@pytest.mark.slow
def test_hd_second_moment_is_one_per_entry():
    """The contract every strategy relies on to keep sigma meaningful.

    Averaged over many blocks rather than within one: inside a block the row norms are fixed
    by construction, so a single-block average would hide a wrong normalization constant
    that is shared across the block.
    """
    d = 64
    m = rows(OrthogonalHD(), jnp.arange(64 * d), d)
    assert abs(float(jnp.mean(jnp.square(m))) - 1.0) < 0.02
    assert abs(float(jnp.mean(m))) < 0.02


@pytest.mark.slow
def test_hd_is_uncorrelated_across_members_within_a_block():
    """Orthogonal directions, and yet entry-wise uncorrelated. Both, deliberately.

    `E[eps_ij eps_i'j] = delta_ii'` holds inside a block: expanding row p of H D1 H D2 H D3
    and taking the D-averages leaves `sum_k H_pk H_p'k / d^2 = delta_pp' / d`. So coupling
    changes neither `E[eps eps^T]` nor the pairwise cross-moments, which is why the estimator
    stays exactly unbiased, and also why the variance of a *linear* functional of the
    population is unchanged.

    Any leverage has to come from the higher-order joint structure that the fitness
    nonlinearity sees. That is not something an argument settles, and it is the reason
    Gate G0 is a measurement rather than a derivation.
    """
    d, blocks = 32, 400
    m = rows(OrthogonalHD(), jnp.arange(d * blocks), d).reshape(blocks, d, d)
    corr = jnp.mean(m[:, 0, :] * m[:, 1, :])
    assert abs(float(corr)) < 0.02, f"members within a block are correlated: {corr}"


def test_hd_pads_and_truncates_a_non_power_of_two():
    """Marginals stay exact; orthogonality degrades. Correctness does not depend on d.

    A slice of an orthogonal matrix's rows is not orthogonal, so this asserts the weaker
    property on purpose: entries still have second moment 1, which is all unbiasedness
    needs, and the Gram is merely close to diagonal.
    """
    d = 40  # pads to 64, so 24 of every 64 design coordinates are discarded
    m = rows(OrthogonalHD(), jnp.arange(64 * 40), d)
    assert m.shape == (64 * 40, d)
    assert abs(float(jnp.mean(jnp.square(m))) - 1.0) < 0.02

    block = rows(OrthogonalHD(), jnp.arange(64), d)
    off = jnp.abs(block @ block.T / d - jnp.eye(64))
    assert float(jnp.max(off)) > 1e-4, "truncation cannot leave the block exactly orthogonal"
    assert float(jnp.mean(off)) < 0.25, "truncation should degrade coupling, not destroy it"


@pytest.mark.parametrize("factors", [1, 2, 3, 5])
def test_hd_is_orthogonal_and_unit_scale_for_any_factor_count(factors):
    """The n^((f-1)/2) normalization, pinned across f.

    f=1 is a bare `H D`, already exactly orthogonal; the extra factors buy closeness to Haar,
    not orthogonality. A wrong exponent shows up here and nowhere else, because every other
    test uses the default.
    """
    d = 64
    m = rows(OrthogonalHD(factors=factors), jnp.arange(d), d)
    assert abs(float(jnp.mean(jnp.square(m))) - 1.0) < 0.05
    gram = (m @ m.T) / d
    assert float(jnp.max(jnp.abs(gram - jnp.eye(d)))) < 1e-4


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_a_nonsense_factor_count(bad):
    with pytest.raises(ValueError, match="at least one HD factor"):
        OrthogonalHD(factors=bad)


# --------------------------------------------------------------------------------------
# ScrambledSobol.
# --------------------------------------------------------------------------------------


def test_sobol_matches_scipy_at_scattered_indices():
    """The skip-ahead test, and the guard on a private scipy API in one.

    Compared as integers, so it is exact rather than to a tolerance: the direct formulation
    is a XOR of direction vectors and there is nothing to round. Index 63 is computed without
    computing 0..62, which is the whole property that makes QMC parallelize (docs/04 C3.2).

    `_direction_numbers` reaches into `scipy.stats._sobol._initialize_v`. A rename breaks the
    import loudly; a change in the numbers would be silent, and this is what catches it.
    """
    from scipy.stats import qmc  # noqa: PLC0415

    d = 5
    ref = qmc.Sobol(d=d, scramble=False, bits=_SOBOL_BITS).random(64)
    v = jnp.asarray(_direction_numbers(d))
    zero = jnp.zeros((d,), jnp.uint32)

    for i in (0, 1, 2, 7, 13, 31, 32, 63):
        got = np.asarray(sobol_point(v, jnp.int32(i), zero))
        want = np.round(ref[i] * 2.0**_SOBOL_BITS).astype(np.uint32)
        assert np.array_equal(got, want), f"point {i}: {got} != {want}"


@pytest.mark.parametrize("d", [1, 3, 16])
def test_sobol_is_a_one_dimensional_net_exactly(d):
    """The low-discrepancy property, asserted exactly rather than statistically.

    Each coordinate of a Sobol sequence is a (0,1)-sequence in base 2, so the first 2^m points
    hit each of the 2^m equal-width bins exactly once. A digital shift permutes the bins and
    preserves that. So the bin indices of N = 2^m members must be a permutation of range(N),
    not merely spread out, and this fails on any iid draw (see the next test).
    """
    m = 8
    n = 1 << m
    v, zero = jnp.asarray(_direction_numbers(d)), jnp.zeros((d,), jnp.uint32)
    x = jax.jit(jax.vmap(lambda i: sobol_point(v, i, zero)))(jnp.arange(n))
    bins = np.asarray(x >> (_SOBOL_BITS - m))
    for j in range(d):
        assert sorted(bins[:, j].tolist()) == list(range(n)), f"dim {j} is not equidistributed"


def test_an_iid_draw_fails_the_net_test():
    """The contrast that makes the test above mean something.

    Without it, `test_sobol_is_a_one_dimensional_net_exactly` could be passing on a bug that
    happens to produce a permutation, and there would be nothing on record saying that iid
    sampling does not.
    """
    m = 8
    n = 1 << m
    u = jax.random.uniform(jax.random.key(7), (n,), dtype=jnp.float32)
    bins = np.asarray((u * n).astype(jnp.int32))
    assert sorted(bins.tolist()) != list(range(n))


@pytest.mark.slow
def test_sobol_first_two_moments():
    """docs/01 C0.5. Mean 0, unit second moment, uncorrelated across dimensions.

    Averaged over shifts as well as members: at a fixed shift the point set is balanced by
    construction, so a single-shift average would pass with a wrong normalization.
    """
    d, n, shifts = 8, 512, 32
    s = ScrambledSobol()
    draws = jnp.stack(
        [rows(s, jnp.arange(n), d, stream=jax.random.fold_in(STREAM, t)) for t in range(shifts)]
    ).reshape(-1, d)

    assert float(jnp.max(jnp.abs(jnp.mean(draws, axis=0)))) < 0.05
    assert float(jnp.max(jnp.abs(jnp.mean(jnp.square(draws), axis=0) - 1.0))) < 0.05

    cov = (draws.T @ draws) / draws.shape[0]
    off = cov - jnp.diag(jnp.diag(cov))
    assert float(jnp.max(jnp.abs(off))) < 0.05


def test_the_digital_shift_is_the_only_randomness():
    """Why scrambling is mandatory, pinned as a mechanism rather than as a tolerance.

    Unscrambled, two different streams give the *same* point set, so the estimator is a fixed
    vector: there is no distribution for a mean over replicates to converge to, and an
    unbiasedness test on it is measuring a constant. Scrambled, the streams differ.

    Member 0 unscrambled is the concrete tell. gray(0) = 0, so its integer point is the origin
    of the unit cube, and every coordinate maps to the same extreme negative value: one point
    on the diagonal at the far corner, which is visibly not a Gaussian draw. Scrambling is what
    turns that corner into a uniformly random one.
    """
    d = 16
    fixed, shifted = ScrambledSobol(scramble=False), ScrambledSobol()
    k1, k2 = jax.random.split(jax.random.key(11))

    assert jnp.array_equal(rows(fixed, jnp.arange(4), d, k1), rows(fixed, jnp.arange(4), d, k2))
    assert not jnp.allclose(
        rows(shifted, jnp.arange(4), d, k1), rows(shifted, jnp.arange(4), d, k2)
    )

    corner = rows(fixed, jnp.array([0]), d, k1)[0]
    assert jnp.all(corner == corner[0]), "unscrambled member 0 is not on the diagonal"
    assert float(corner[0]) < -5.0, f"expected the far corner, got {corner[0]}"


@pytest.mark.parametrize("d", [None, 512 * 512], ids=["one_past_maxdim", "full_rank_leaf"])
def test_sobol_refuses_a_dimension_past_the_table(d):
    """The second, sharper reason coupling is low-rank only: at d = mn the scheme is not
    constructible, not merely unhelpful (docs/01 C0.5).

    512x512 is the Phase 0 transformer leaf, so the full-rank case is the one that would
    actually be reached. `registry.check_entry` refuses that cell first, but a hand-built
    strategy has to fail loudly rather than silently sample something else.
    """
    from scipy.stats import qmc  # noqa: PLC0415

    with pytest.raises(ValueError, match="Joe-Kuo table stops"):
        ScrambledSobol()(STREAM, jnp.int32(0), d or qmc.Sobol.MAXDIM + 1, jnp.float32)


# --------------------------------------------------------------------------------------
# The seed contract, at the coupling level.
# --------------------------------------------------------------------------------------

couplings = pytest.mark.parametrize(
    "coupling",
    [Gaussian(), OrthogonalHD(), ScrambledSobol()],
    ids=["iid", "orthogonal_hd", "sobol"],
)


@couplings
def test_a_members_draw_depends_only_on_its_global_id(coupling):
    """Invariant 2, one level below where test_strategies.py checks it.

    Checked here as well as there because a coupling is the one place a *positional* index
    could creep in: the block is `i // d` and the position `i % d`, both functions of the id
    alone. A coupling that used the batch position instead would still pass every
    orthogonality test above.
    """
    d = 16
    alone = coupling(STREAM, jnp.int32(37), d, jnp.float32)
    batched = rows(coupling, jnp.array([37, 4, 300]), d)[0]
    shifted = rows(coupling, jnp.arange(30, 40), d)[7]
    assert jnp.allclose(alone, batched, atol=1e-6)
    assert jnp.allclose(alone, shifted, atol=1e-6)


@couplings
def test_a_coupling_survives_two_separate_jit_traces(coupling):
    """Caught a real bug, and only under jit.

    `_direction_numbers` is memoized, and it used to return a `jnp` array. The first call
    happened inside whatever trace reached it first, so the cache stored a tracer and every
    later trace got a leaked one. Eager tests could never see it: there was no second trace.

    Anything a coupling memoizes has to be host data. Two traces with different `d` and
    different streams is the cheapest thing that would have failed.
    """
    a = rows(coupling, jnp.arange(4), 8, stream=jax.random.key(31))
    b = rows(coupling, jnp.arange(4), 16, stream=jax.random.key(32))
    assert a.shape == (4, 8) and b.shape == (4, 16)


@couplings
def test_independent_streams_give_independent_draws(coupling):
    """Two streams are two design families. LowRank leans on this for its r columns and for
    keeping A's design separate from B's."""
    d = 256
    k1, k2 = jax.random.split(jax.random.key(5))
    a = rows(coupling, jnp.arange(4), d, stream=k1)
    b = rows(coupling, jnp.arange(4), d, stream=k2)
    assert float(jnp.max(jnp.abs(jnp.sum(a * b, axis=1) / d))) < 8 / jnp.sqrt(d)


def test_gaussian_singleton_is_stateless():
    """`GAUSSIAN` is a module-level instance because couplings hold no state. If one ever
    grows state, this is what stops it being shared by accident."""
    assert isinstance(GAUSSIAN, Gaussian)
    assert not vars(GAUSSIAN)


@pytest.mark.parametrize(
    "coupling",
    [Gaussian(), OrthogonalHD(), ScrambledSobol(), GAUSSIAN],
    ids=["iid", "orthogonal_hd", "sobol", "singleton"],
)
def test_conforms_to_the_coupling_protocol(coupling):
    assert isinstance(coupling, Coupling)


# --------------------------------------------------------------------------------------
# Where the coupling lands inside a strategy.
# --------------------------------------------------------------------------------------


def test_passing_the_default_coupling_changes_nothing():
    """docs/04's `test_coupled_reduces_to_iid`. Bitwise, not to tolerance.

    The uncoupled path has to stay exactly what it was, or every baseline in the sweep moves
    when coupling is added and the comparison is against a different algorithm.
    """
    params = {"w": jnp.ones((6, 4), dtype=jnp.float32), "b": jnp.ones((4,), dtype=jnp.float32)}
    key, ids = jax.random.key(1), jnp.arange(8)
    w = jax.random.normal(jax.random.key(2), (8,), dtype=jnp.float32)

    for default, explicit in [
        (IIDGaussian(), IIDGaussian(coupling=Gaussian())),
        (SeedRegenerated(), SeedRegenerated(coupling=GAUSSIAN)),
        (LowRank(r=2), LowRank(r=2, coupling=Gaussian())),
    ]:
        a = default.contract(sampled(default, key, params, ids), w)
        b = explicit.contract(sampled(explicit, key, params, ids), w)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert jnp.array_equal(x, y), type(default).__name__


def test_coupling_lands_on_the_lowrank_factors_not_the_product():
    """The design axis under low rank is `a` in R^m, not `E` in R^(mk).

    Getting this wrong is the plausible bug: coupling the product would leave the factors
    iid, still give an unbiased estimator, and sample in the space the whole EGGROLL argument
    says is the wrong one. So it is asserted where it is visible, on the factors themselves.
    """
    m, k = 16, 5
    params = {"w": jnp.zeros((m, k), dtype=jnp.float32)}
    s = LowRank(r=1, coupling=OrthogonalHD())
    pert = sampled(s, jax.random.key(3), params, jnp.arange(m))  # exactly one block

    a = pert.factors["w"].a[:, :, 0]
    assert float(jnp.max(jnp.abs(a @ a.T / m - jnp.eye(m)))) < 1e-4

    # k=5 pads to 8, so the b-side is truncated and only near-orthogonal. Asserted loosely
    # on purpose: the point is that b is coupled at all, in its own space of dimension k.
    b = pert.factors["w"].b[:, :, 0]
    assert abs(float(jnp.mean(jnp.square(b))) - 1.0) < 0.2


def test_lowrank_columns_are_separate_design_families():
    """r columns, r families. Sharing one stream across columns would make every column of a
    member's A identical, which is rank 1 wearing a rank-r shape."""
    m, r = 32, 4
    params = {"w": jnp.zeros((m, 6), dtype=jnp.float32)}
    s = LowRank(r=r, coupling=OrthogonalHD())
    a = sampled(s, jax.random.key(4), params, jnp.arange(m)).factors["w"].a

    for j in range(1, r):
        assert not jnp.allclose(a[:, :, 0], a[:, :, j]), f"column {j} duplicates column 0"
    # Each column is orthogonal within the block in its own right.
    for j in range(r):
        col = a[:, :, j]
        assert float(jnp.max(jnp.abs(col @ col.T / m - jnp.eye(m)))) < 1e-4


def test_sobol_streams_get_different_direction_numbers():
    """BACKLOG B1's fix, asserted at the mechanism rather than through the estimator.

    A digital shift cancels between two members of the same stream:

        (x_i XOR s) XOR (x_j XOR s) = x_i XOR x_j

    so with one shared block of direction numbers the inter-member XOR geometry is identical
    in *every* stream, and a deficiency in that one arrangement adds coherently across leaves
    instead of averaging over independent draws. Measured cost was 5% of i.i.d. cosine at 16
    streams (experiments/phase1/sobol_b1.py).

    The property that fixes it is that different streams draw *different* direction numbers,
    so their XOR geometries genuinely differ. Checked here on the raw uniform points, before
    ndtri, because that is where the claim lives.
    """
    from shardes.coupling import _SOBOL_BITS, _direction_numbers, sobol_point

    d, ids = 32, jnp.arange(8)

    def geometry(coupling, stream):
        """x_i XOR x_j for the first two members, which the shift cannot change."""
        k_shift, k_block = jax.random.split(stream)
        v = jnp.asarray(_direction_numbers(coupling._span(d)))
        if coupling.blocks > 1:
            b = jax.random.randint(k_block, (), 0, coupling._blocks(d))
            v = jax.lax.dynamic_slice_in_dim(v, b * d, d, axis=1)
        shift = jax.random.bits(k_shift, (d,), jnp.uint32) >> (32 - _SOBOL_BITS)
        pts = jnp.stack([sobol_point(v, i, shift) for i in ids])
        # Every member against member 0, not just members 0 and 1. Members 0 and 1 differ by
        # exactly v[0], and Sobol's first direction vector is 2^29 in *every* dimension, so
        # that one pair is identical across blocks by construction and detects nothing. The
        # first version of this test used it and failed for that reason.
        return pts[1:] ^ pts[0]

    s1, s2 = jax.random.split(jax.random.key(0))

    shared = ScrambledSobol(blocks=1)
    assert jnp.array_equal(geometry(shared, s1), geometry(shared, s2)), (
        "with one block the inter-member geometry must be identical across streams; "
        "if this fails the premise of the B1 finding is wrong"
    )

    blocked = ScrambledSobol(blocks=16)
    assert not jnp.array_equal(geometry(blocked, s1), geometry(blocked, s2)), (
        "with per-stream blocks the geometry must differ across streams"
    )


def test_sobol_blocks_do_not_break_the_seed_contract():
    """The block index comes from `stream`, never from the member, so member i is still
    member i however the population is batched."""
    c = ScrambledSobol(blocks=16)
    a = rows(c, jnp.array([5, 9]), 32)
    b = rows(c, jnp.arange(64), 32)
    assert jnp.allclose(a[0], b[5], atol=1e-6) and jnp.allclose(a[1], b[9], atol=1e-6)
