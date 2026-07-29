"""The Phase 0 transformer block and its backprop oracle.

Small `d_model` throughout: the block's correctness does not depend on width, and the
suite has a two-minute budget. The 512-wide version is what the sweep runs.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes.dimensions import FULL, sampling_dimension
from shardes.problems import transformer_block as tb

D = 16


@pytest.fixture
def setup():
    params = tb.init(jax.random.key(0), d_model=D)
    batch = tb.make_batch(jax.random.key(1), d_model=D, batch=4, seq=8)
    return params, batch


def test_forward_preserves_shape(setup):
    params, batch = setup
    assert tb.forward(params, batch.x).shape == batch.x.shape


def test_every_parameter_is_a_square_matrix(setup):
    """d_ff = d_model, so d_eff is one number rather than a mixture. Not a realistic
    transformer ratio; see the module docstring and the limitations section."""
    params, _ = setup
    assert set(params) == set(tb.NAMES)
    for name, leaf in params.items():
        assert leaf.shape == (D, D), name


def test_loss_is_a_scalar(setup):
    params, batch = setup
    assert tb.loss(params, batch).shape == ()


def test_analytic_grad_matches_numerical(setup):
    """The oracle checked against a genuinely independent method.

    `tb.grad` is backprop, so comparing it to `jax.grad` would compare autodiff with
    itself. Central differences have entirely different failure modes, which is what
    docs/conventions.md asks for when no closed form exists.
    """
    params, batch = setup
    got = tb.grad(params, batch)

    eps = 1e-3
    name = "wo"
    idx = (2, 3)
    up = {k: (v.at[idx].add(eps) if k == name else v) for k, v in params.items()}
    down = {k: (v.at[idx].add(-eps) if k == name else v) for k, v in params.items()}
    numeric = (tb.loss(up, batch) - tb.loss(down, batch)) / (2 * eps)

    assert jnp.isclose(got[name][idx], numeric, rtol=2e-2, atol=1e-6)


def test_grad_tree_matches_params_tree(setup):
    params, batch = setup
    g = tb.grad(params, batch)
    assert jax.tree.structure(g) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(params)):
        assert a.shape == b.shape


def test_grad_is_nonzero(setup):
    """A block that is accidentally an identity would give a zero gradient and every
    cosine downstream would be undefined."""
    params, batch = setup
    assert float(jnp.linalg.norm(jax.tree.leaves(tb.grad(params, batch))[0])) > 0


def test_deterministic():
    a = tb.init(jax.random.key(5), d_model=D)
    b = tb.init(jax.random.key(5), d_model=D)
    assert all(jnp.array_equal(a[k], b[k]) for k in a)


def test_forward_is_not_a_no_op(setup):
    """Residual-only would make the block invisible to the perturbation."""
    params, batch = setup
    assert not jnp.allclose(tb.forward(params, batch.x), batch.x)


def test_activations_stay_finite_at_width(setup):
    """The 1/sqrt(d) init exists so a 512-wide block does not blow up on the real run."""
    params = tb.init(jax.random.key(6), d_model=256)
    batch = tb.make_batch(jax.random.key(7), d_model=256, batch=2, seq=4)
    out = tb.forward(params, batch.x)
    assert jnp.all(jnp.isfinite(out))
    assert float(jnp.max(jnp.abs(out))) < 1e3


def test_d_eff_at_the_sweep_width():
    """The numbers quoted in docs/01 C0.4, from the real params tree."""
    params = tb.init(jax.random.key(8), d_model=512)
    assert sampling_dimension(params, FULL) == 1_572_864
    assert sampling_dimension(params, 1) == 6_144
