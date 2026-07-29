"""Estimator-quality metrics, against hand-computable cases.

These are what every Phase 0 number is reported through, so a quiet error here would not
show up as a failure anywhere. It would show up as a plot.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shardes import metrics

RTOL = 1e-6


@pytest.fixture
def tree():
    k1, k2 = jax.random.split(jax.random.key(0))
    return {
        "w": jax.random.normal(k1, (4, 3), dtype=jnp.float32),
        "b": jax.random.normal(k2, (3,), dtype=jnp.float32),
    }


def test_vdot_matches_numpy_on_a_flat_case():
    x = jnp.arange(5.0)
    y = jnp.arange(5.0) * 2 - 1
    assert jnp.isclose(metrics.tree_vdot(x, y), float(np.dot(np.arange(5.0), np.arange(5.0) * 2 - 1)))


def test_vdot_sums_over_leaves(tree):
    """The multi-leaf sum is what makes this different from a dot product on one array."""
    by_hand = sum(float(jnp.vdot(v, v)) for v in jax.tree.leaves(tree))
    assert jnp.isclose(metrics.tree_vdot(tree, tree), by_hand, rtol=RTOL)


def test_structure_mismatch_raises(tree):
    """zip would truncate silently and return a plausible wrong number."""
    smaller = {"w": tree["w"]}
    with pytest.raises(ValueError, match="structure mismatch"):
        metrics.tree_vdot(tree, smaller)
    with pytest.raises(ValueError, match="structure mismatch"):
        metrics.relative_mse(tree, smaller)


def test_empty_tree_raises():
    with pytest.raises(ValueError, match="empty pytree"):
        metrics.tree_vdot({}, {})


def test_cosine_of_a_vector_with_itself(tree):
    assert jnp.isclose(metrics.cosine_similarity(tree, tree), 1.0, rtol=RTOL)


def test_cosine_of_a_vector_with_its_negation(tree):
    neg = jax.tree.map(lambda x: -x, tree)
    assert jnp.isclose(metrics.cosine_similarity(neg, tree), -1.0, rtol=RTOL)


def test_cosine_is_scale_free(tree):
    """The reason cosine is the headline metric: ES applies a learning rate anyway."""
    base = metrics.cosine_similarity(tree, tree)
    for a in (0.01, 3.0, 1000.0):
        scaled = jax.tree.map(lambda x: a * x, tree)
        assert jnp.isclose(metrics.cosine_similarity(scaled, tree), base, rtol=RTOL)


def test_cosine_of_orthogonal_vectors():
    a = {"x": jnp.array([1.0, 0.0]), "y": jnp.array([0.0])}
    b = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0])}
    assert jnp.isclose(metrics.cosine_similarity(a, b), 0.0, atol=1e-7)


def test_relative_mse_of_identical_is_zero(tree):
    assert float(metrics.relative_mse(tree, tree)) == 0.0


@pytest.mark.parametrize("factor,want", [(2.0, 1.0), (3.0, 4.0), (0.5, 0.25), (-1.0, 4.0)])
def test_relative_mse_known_value(factor, want):
    """g_hat = a*grad gives (a-1)^2.

    a = 2 is degenerate and must not be the only case: it gives 1 whether or not the
    difference is squared, so a test built on it alone passes against a broken
    ||diff||/||grad||. Mutation testing found exactly that, which is why the other three
    factors are here.
    """
    grad = {"x": jnp.array([3.0, 4.0])}
    g_hat = jax.tree.map(lambda x: factor * x, grad)
    assert jnp.isclose(metrics.relative_mse(g_hat, grad), want, rtol=RTOL)


def test_relative_bias_is_zero_when_the_mean_is_exact(tree):
    assert float(metrics.relative_bias(tree, tree)) == 0.0


def test_relative_bias_known_value():
    grad = {"x": jnp.array([3.0, 4.0])}  # norm 5
    off = {"x": jnp.array([3.0, 5.0])}  # differs by 1
    assert jnp.isclose(metrics.relative_bias(off, grad), 0.2, rtol=RTOL)


def test_bf16_accumulates_in_f32():
    """f32 accumulation over a long bf16 reduction, against an exact f64 reference.

    XLA reduces pairwise rather than sequentially, so bf16 is nowhere near as bad as a
    naive running total would be. Do not reach for identical inputs to demonstrate this:
    summing n ones in bf16 is *exact*, because every pairwise partial is a power of two.

    Measured at n = 2^18 on random normals: bf16 accumulation is 1.2e-3 relative error,
    f32 accumulation is 5.4e-8. Four orders of magnitude, which is what the rule in
    docs/conventions.md is buying.
    """
    n = 1 << 18
    x = jax.random.normal(jax.random.key(11), (n,), dtype=jnp.float32).astype(jnp.bfloat16)

    exact = np.asarray(x, dtype=np.float64)  # bf16 values are exact in f64
    ref = float(exact @ exact)

    ours = abs(float(metrics.tree_vdot(x, x)) - ref) / ref
    naive = abs(float(jnp.vdot(x, x, preferred_element_type=jnp.bfloat16)) - ref) / ref

    assert ours < 1e-6, f"f32 accumulation drifted: {ours:.2e}"
    assert naive > 1e-4, f"bf16 accumulation is no longer a hazard ({naive:.2e}), recheck"


def test_jit(tree):
    assert jnp.isclose(
        jax.jit(metrics.cosine_similarity)(tree, tree),
        metrics.cosine_similarity(tree, tree),
        rtol=RTOL,
    )
