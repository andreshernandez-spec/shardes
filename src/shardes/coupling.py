"""Coupling: how one member's noise relates to the rest of the population's.

A `Coupling` replaces "draw d iid normals for member i" with "give member i its share of a
point set designed across members". `Gaussian` is the uncoupled default and is exactly what
the strategies did before this module existed.

**This is not the wrapper docs/04 C3.1 predicted, and the difference is the finding.**

`Mirrored` gets to be a wrapper because antithetic sampling only touches *signs*: it reuses
the inner perturbation untouched and folds the sign into `contract`'s weights, so the inner
stays opaque. Coupling changes the *directions*, and a wrapper would have to reach inside
the inner perturbation and replace the noise it had just drawn. Three separate things break
there:

- The perturbation is opaque by protocol and only the strategy knows which of its arrays
  are design axes. LowRank's are the (m, r) and (k, r) factors, never the (m, k) product;
  IIDGaussian's is the flattened leaf. A wrapper would have to know both layouts, and the
  next strategy's too.
- SeedRegenerated materializes nothing. There is no array to reach into, by design.
- HD coupling does not *transform* iid noise, it *replaces* it. Rows of H D1 H D2 H D3 are
  built from Rademacher signs and consume no Gaussian, so drawing one first is wasted work
  rather than an input.

So coupling is a constructor argument on the strategy, not a strategy wrapped around it:

    IIDGaussian(coupling=OrthogonalHD())
    Mirrored(LowRank(r=1, coupling=OrthogonalHD()))

The substance of C3.1's check still holds: nothing in the sharded core changes, and what
moved is one line inside each `sample`. The seed contract survives because the block index
is `member_id // d` and the position within it is `member_id % d`, both functions of the
global id alone, so the point set does not depend on how members are batched or sharded.

The signature is scalar in the member on purpose. `sample` already runs under vmap
(IIDGaussian, LowRank) or scan (SeedRegenerated) over members, and a per-member draw is the
only form that works unchanged in both.
"""

import functools
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from shardes.transforms.fwht import fwht
from shardes.types import Array, Key


@runtime_checkable
class Coupling(Protocol):
    """A per-member draw with per-entry second moment 1."""

    def __call__(self, stream: Key, member_id: Array, d: int, dtype) -> Array:
        """(d,) for one member.

        `stream` names an independent family and must not depend on `member_id`: it is
        precisely what the members of a family share and design against. `member_id` is the
        GLOBAL index. `d` is static, so a coupling may branch on it.

        Unit per-entry second moment is the contract every strategy relies on to keep sigma
        meaningful, and it must hold marginally for each member, not just on average over
        the population.
        """
        ...


class Gaussian:
    """No coupling. Member i's draw is independent of every other member's.

    The default everywhere, and the baseline the coupled schemes are measured against.
    """

    def __call__(self, stream: Key, member_id: Array, d: int, dtype) -> Array:
        return jax.random.normal(jax.random.fold_in(stream, member_id), (d,), dtype)


def _next_pow2(d: int) -> int:
    return 1 << max(d - 1, 0).bit_length()


def hadamard_row(p: Array, n: int) -> Array:
    """Row p of the n x n Sylvester Hadamard matrix, in O(n).

    Equal to fwht(one_hot(p)) and asserted against it, but it skips a whole butterfly:
    H[p, j] = (-1)^popcount(p & j) straight from the Sylvester recursion. Worth the three
    lines because the HD product starts from a one-hot every time, so this is 1/factors of
    the total cost.
    """
    j = jnp.arange(n, dtype=jnp.uint32)
    parity = jax.lax.population_count(j & p.astype(jnp.uint32)) & jnp.uint32(1)
    return 1.0 - 2.0 * parity.astype(jnp.float32)


class OrthogonalHD:
    """Row `i % d` of an orthogonal `H D1 H D2 H D3`, from block `i // d`.

    `H/sqrt(d)` is orthogonal and so is every Rademacher diagonal `D`, so the product is
    **exactly** orthogonal, not approximately. Within a block the members' directions are
    orthonormal to float precision. What is only approximate is the *distribution*: rows are
    near-uniform on the sphere rather than Haar. docs/04 C3.1 had this the wrong way round
    and is corrected there; `test_hd_block_is_exactly_orthogonal` pins the strong version.

    Blocks are independent, so members in different blocks are near-orthogonal by
    concentration at `O(1/sqrt(d))` rather than exactly. That is the property that makes
    this the good case under sharding: a device holding a whole number of blocks needs no
    communication to be coupled (docs/04 C3.2).

    Unbiasedness is exact, and cheaply so. `D_f` applies an independent sign to every
    coordinate last, so the draw is symmetric coordinate-wise and every odd moment
    vanishes; and `E[(H D1 H D2 H D3)_pj^2] = 1/d` exactly by expanding the square and using
    `E[d_k d_k'] = delta_kk'`, so `E[eps eps^T] = I`. The quadratic oracle in
    test_estimator.py stays exact under coupling, which is why coupled strategies sit in the
    same unbiasedness suite as the rest rather than in a special case.

    Cost is `O(d log d)` per member and `O(d)` memory. QR of a d x d Gaussian would give
    Haar exactly and costs `O(d^3)`, which is not available at d = 262144.

    **Non-power-of-two d is padded and truncated.** The FWHT needs a power of two. Padding
    to `d' >= d` and keeping the first d entries leaves the marginals exact (the identity
    above is per-entry), so unbiasedness is untouched; what degrades is orthogonality,
    because a slice of an orthogonal matrix's rows is no longer orthogonal. Coupling quality
    falls off gracefully, correctness does not. Worst case is `d` just over a power of two,
    where half the design is discarded.
    """

    def __init__(self, factors: int = 3):
        if factors < 1:
            raise ValueError(f"OrthogonalHD needs at least one HD factor, got {factors}")
        self.factors = int(factors)

    def __call__(self, stream: Key, member_id: Array, d: int, dtype) -> Array:
        n = _next_pow2(d)
        keys = jax.random.split(jax.random.fold_in(stream, member_id // n), self.factors)

        v = hadamard_row(member_id % n, n) * jax.random.rademacher(
            keys[0], (n,), dtype=jnp.float32
        )
        # Indexed rather than iterated: `keys` is a traced key array under vmap and a Python
        # `for k in keys` leans on __iter__ working there.
        for j in range(1, self.factors):
            v = fwht(v) * jax.random.rademacher(keys[j], (n,), dtype=jnp.float32)

        # f unnormalized H's give E[v_j^2] = n^(f-1); this brings it to 1, which also makes
        # the row norm sqrt(n) and matches what a Gaussian draw would deliver.
        return (v[:d] / n ** ((self.factors - 1) / 2)).astype(dtype)


# Sobol. 30 bits is scipy's default, so an unscrambled draw is comparable to
# `scipy.stats.qmc.Sobol` point for point and skip-ahead is testable against it.
_SOBOL_BITS = 30

# f32 represents `k + 0.5` exactly only below 2^23, and `u == 1.0` would send ndtri to +inf.
# Taking the top 22 bits keeps `u` strictly inside (0, 1) and exactly symmetric about 0.5,
# which is what makes E[eps] exactly zero. Truncating low bits of a digital net gives the
# coarser net exactly (XOR is bitwise, so it commutes with a right shift), so this costs
# resolution and nothing else. It caps the usable population at 2^22, well past the 2^18 the
# sweep reaches, and the low-discrepancy structure at N points only needs log2(N) bits.
_UNIFORM_BITS = 22


@functools.lru_cache(maxsize=None)
def _direction_numbers(d: int) -> "np.ndarray":  # noqa: F821
    """Joe-Kuo direction numbers for `d` dimensions, as a (bits, d) uint32 **numpy** array.

    Pulled from scipy rather than vendored. The table is the scarce artifact here, not the
    algorithm, and scipy carries Joe-Kuo to 21,201 dimensions
    (docs/04-phase3-coupling.md C3.2).

    **Numpy, and cached as numpy.** Returning a `jnp` array from an `lru_cache` is a real bug
    and it bit: the first call happens inside whatever trace reaches it first, `jnp.asarray`
    emits a primitive there, and the cache then hands a leaked tracer to every later trace.
    Host data in the cache, device conversion at the call site.
    `test_a_coupling_survives_two_separate_jit_traces` is what catches a regression.

    `scipy.stats._sobol._initialize_v` is private, and that is a real fragility rather than a
    tidy one: a scipy rename breaks the import loudly, and a change in the *numbers* would be
    silent. `test_sobol_matches_scipy_at_scattered_indices` is the guard, comparing against the
    public `qmc.Sobol`. Both live or both die.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.stats import _sobol, qmc  # noqa: PLC0415

    if d > qmc.Sobol.MAXDIM:
        raise ValueError(
            f"sobol needs direction numbers for d={d}, and the Joe-Kuo table stops at "
            f"{qmc.Sobol.MAXDIM}. This is why the sweep grid has no full-rank sobol cell: "
            "full-rank sampling is in R^(mn). docs/01-phase0-estimator-harness.md C0.5."
        )
    v = np.zeros((d, _SOBOL_BITS), dtype=np.uint32)
    _sobol._initialize_v(v, d, _SOBOL_BITS)
    return v.T.copy()  # (bits, d), so v[k] is one bit's direction vector


def sobol_point(v: Array, member_id: Array, shift: Array) -> Array:
    """Point `member_id` of a digitally shifted digital net, as (d,) uint32.

    The XOR of the direction vectors selected by the set bits of `gray(i)`. Split out from
    `ScrambledSobol.__call__` so the integer point can be compared to scipy exactly, rather
    than through a float round-trip that would need a tolerance.
    """
    i = member_id.astype(jnp.uint32)
    gray = i ^ (i >> 1)
    x = shift
    # One (d,) buffer live at a time. Masking all `bits` rows at once and XOR-reducing would
    # be log-depth instead of linear, but under vmap it materializes (n, bits, d): 16 GB at
    # n = 2^18, d = 512, against 0.5 GB for the loop.
    for k in range(_SOBOL_BITS):
        x = jnp.where((gray >> k) & 1 == 1, x ^ v[k], x)
    return x


class ScrambledSobol:
    """Member i is point i of a randomly digitally shifted Sobol sequence, through Phi^-1.

    The direct formulation: point i is the XOR of the direction vectors selected by the set
    bits of `gray(i) = i ^ (i >> 1)`. Random-access by construction, so skip-ahead across
    devices is not a feature to add, it is the only thing this does. The sequential Gray-code
    recurrence that most reference implementations use (including NVIDIA's CUDA sample) is
    faster for emitting points 0..N in order and is the reason skip-ahead has a reputation for
    being hard.

    **Scrambling is mandatory, not optional.** A deterministic Sobol point set has no
    randomness at all, so the estimator is a fixed vector and there is nothing for the mean
    over replicates to converge to; the estimate feeds SGD, so unbiasedness is not something
    to trade. A random digital shift is enough (one uniform integer per dimension, XORed in)
    and full Owen nesting is not needed. The shift comes from `stream`, which is shared across
    members and devices, and the point index is the global member id, so device-count
    invariance falls out of the construction rather than being tested for afterwards.

    `scramble=False` exists for two tests and no other reason: comparing against scipy's
    sequence, and pinning that the deterministic version really is degenerate.

    **Low-rank paths only.** The design dimension has to be under 21,201, which a factor axis
    (m or k) is and a whole leaf (mk) is not. `_direction_numbers` raises on the latter, and
    `registry.check_entry` refuses the cell before it gets that far.

    **Two reasons to expect this arm to underperform, written down before the run.** Recording
    them now so a null result reads as a prediction rather than an excuse:

    - Sobol's guarantee rests on **low effective dimension**, and `f(theta + sigma eps)`
      depends on every coordinate of eps roughly equally. High effective dimension is the
      worst case for QMC, and it is the case ES is in by construction.
    - Sobol's **2-D projections degrade badly in the later dimensions**. At m = 512 the design
      is a 512-dimensional point set, well into the range where that is a known weakness of
      Joe-Kuo tables rather than a subtlety. `orthogonal_hd` has no analogue of this, which is
      part of why docs/01 C0.5 puts it in both rank panels and makes it carry the comparison.

    Neither is a reason not to measure it. They are reasons the measurement is the point.

    **`blocks`: each stream draws a different block of Sobol dimensions, and this is a fix for
    a measured defect rather than a tuning knob.** E1 found this scheme systematically worse
    than uncoupled sampling, degrading with N to -11% at N = 2^18. `experiments/phase1/
    sobol_b1.py` identified the cause and `docs/BACKLOG.md` B1 records it.

    A digital shift *translates* a point set without changing its geometry: for two members
    within a stream, `(x_i XOR s) XOR (x_j XOR s) = x_i XOR x_j`, so the shift cancels and the
    inter-member arrangement is identical in **every** stream, where i.i.d. gives each an
    independent one. A deficiency in that single shared arrangement is then added coherently
    across every leaf and factor axis instead of being averaged over independent draws.

    Measured: the penalty grows with the number of streams (0.984 of i.i.d. at 2 streams,
    0.950 at 16), giving each stream its own block of dimensions recovers it to 0.99, and a
    control that permutes *coordinates* rather than dimensions does **not** help — which
    confirms the mechanism rather than contradicting it, since a consistent permutation
    preserves `x_i XOR x_j` and only relabels it.

    The residual ~1% is not a defect to chase: it is the two a-priori reasons above.

    `blocks=1` restores the old single-block behaviour, which is what E1's sobol arm measured.
    """

    def __init__(self, scramble: bool = True, blocks: int = 16):
        self.scramble = bool(scramble)
        self.blocks = max(int(blocks), 1)

    def __call__(self, stream: Key, member_id: Array, d: int, dtype) -> Array:
        k_shift, k_block = jax.random.split(stream)
        shift = (
            jax.random.bits(k_shift, (d,), jnp.uint32) >> (32 - _SOBOL_BITS)
            if self.scramble
            else jnp.zeros((d,), jnp.uint32)
        )
        x = sobol_point(self.directions(k_block, d), member_id, shift)
        u = (x >> (_SOBOL_BITS - _UNIFORM_BITS)).astype(jnp.float32)
        return jax.scipy.special.ndtri((u + 0.5) * 2.0**-_UNIFORM_BITS).astype(dtype)

    def directions(self, k_block: Key, d: int) -> Array:
        """This stream's (bits, d) direction numbers.

        Public so a test can exercise the block choice through the real code rather than
        restating it. That is not a stylistic preference: the first version of
        `test_sobol_streams_get_different_direction_numbers` reimplemented this selection
        inline, so it compared its own copy against itself and a mutation that made every
        stream draw block 0 survived unnoticed. `experiments/mutation.py` found it.
        """
        v = jnp.asarray(_direction_numbers(self._span(d)))
        if self.scramble and self.blocks > 1:
            b = jax.random.randint(k_block, (), 0, self._blocks(d))
            v = jax.lax.dynamic_slice_in_dim(v, b * d, d, axis=1)
        return v

    def _blocks(self, d: int) -> int:
        """How many disjoint d-dimension blocks the Joe-Kuo table affords."""
        from scipy.stats import qmc  # noqa: PLC0415

        return max(min(self.blocks, qmc.Sobol.MAXDIM // max(d, 1)), 1)

    def _span(self, d: int) -> int:
        return d * (self._blocks(d) if self.scramble and self.blocks > 1 else 1)


GAUSSIAN = Gaussian()

BY_NAME = {
    "iid": GAUSSIAN,
    "orthogonal_hd": OrthogonalHD(),
    "sobol": ScrambledSobol(),
}
