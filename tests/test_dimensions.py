"""Sampling dimension, the x-axis of F5.

The numbers here are the ones quoted in docs/01 C0.4, so this file is what keeps the doc
honest if the block ever changes shape.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes.dimensions import FULL, sampling_dimension

D = 512
MATS = 6


@pytest.fixture
def block_params():
    """The Phase 0 block: six square matrices, no learnable norms."""
    return {
        name: jnp.zeros((D, D), dtype=jnp.float32)
        for name in ("wq", "wk", "wv", "wo", "w_up", "w_down")
    }


def test_full_rank_is_the_total_parameter_count(block_params):
    """One member's eps covers every matrix at once, so the sampling space is the
    concatenation, not a single matrix."""
    assert sampling_dimension(block_params, FULL) == MATS * D * D == 1_572_864


def test_rank_one_is_the_sum_of_m_plus_n(block_params):
    assert sampling_dimension(block_params, 1) == MATS * (D + D) == 6_144


def test_rank_four_scales_linearly(block_params):
    assert sampling_dimension(block_params, 4) == 4 * sampling_dimension(block_params, 1)


def test_the_gap_between_panels_is_256x(block_params):
    """The inversion EGGROLL buys, and the reason the two panels are on the same axis."""
    full = sampling_dimension(block_params, FULL)
    rank1 = sampling_dimension(block_params, 1)
    assert full // rank1 == 256


def test_the_sweep_straddles_the_line_where_it_should(block_params):
    """G0 needs full rank to stay well under N/d_eff = 1 and rank 1 to cross it."""
    n_max = 2**18
    assert n_max / sampling_dimension(block_params, FULL) < 0.2
    assert n_max / sampling_dimension(block_params, 1) > 40
    assert 2**6 / sampling_dimension(block_params, 1) < 0.05


def test_vector_leaves_contribute_their_full_size_at_any_rank():
    """A norm scale or bias has no low-rank factorisation. Unused by the Phase 0 block,
    which is all matrices, and load-bearing once mixed leaves arrive in Phase 1."""
    params = {"w": jnp.zeros((8, 4)), "scale": jnp.zeros((8,))}
    assert sampling_dimension(params, FULL) == 32 + 8
    assert sampling_dimension(params, 1) == (8 + 4) + 8


def test_accepts_a_shape_only_tree():
    """The driver computes d_eff from a spec, without building the model."""
    spec = {"w": jax.ShapeDtypeStruct((D, D), jnp.float32)}
    assert sampling_dimension(spec, FULL) == D * D
    assert sampling_dimension(spec, 1) == 2 * D
