"""`check_model`: the authoring-time version of the seam constraint.

A structured strategy substitutes a `LowRankWeight`, so a model must route matmuls through
`dense` and gathers through `embed`. That fails loudly at run time, which is safe but arrives
minutes into a sweep. This gets the same answer in a second, on CPU, and reports every
offending leaf rather than the first.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes.check import check_model
from shardes.nn import dense, embed

PARAMS = {
    "emb": jnp.ones((20, 4), jnp.float32),
    "w1": jnp.ones((8, 4), jnp.float32),
    "w2": jnp.ones((3, 8), jnp.float32),
    "bias": jnp.zeros((8,), jnp.float32),
}
BATCH = jnp.array([[0, 1], [2, 3]])


def good(p, b):
    h = jnp.mean(embed(p["emb"], b), axis=1)
    h = jnp.tanh(dense(h, p["w1"]) + p["bias"])
    return jnp.sum(dense(h, p["w2"]))


def test_a_correct_model_has_no_findings():
    assert check_model(good, PARAMS, BATCH) == []


def test_a_raw_matmul_is_reported_with_its_leaf():
    """The message has to name the leaf. "somewhere in your model there is a matmul" is not
    actionable on a params tree with thirty entries."""
    def bad(p, b):
        h = jnp.mean(embed(p["emb"], b), axis=1)
        return jnp.sum(h @ p["w1"].T)

    found = check_model(bad, PARAMS, BATCH)
    # The bracketed path, not the bare name: the remedy text contains the literal
    # "shardes.nn.embed", so `"emb" in f` matches messages about other leaves entirely.
    assert any("['w1']" in f and "transposition" in f for f in found), found


def test_a_raw_gather_is_reported():
    def bad(p, b):
        h = jnp.mean(p["emb"][b], axis=1)
        return jnp.sum(dense(jnp.tanh(dense(h, p["w1"])), p["w2"]))

    found = check_model(bad, PARAMS, BATCH)
    assert any("['emb']" in f and "indexing" in f for f in found), found


def test_every_offending_leaf_is_reported_not_just_the_first():
    """One leaf at a time, so a model with two mistakes yields two findings. Failing on the
    first and hiding the rest is what the run-time error already does."""
    def bad(p, b):
        h = jnp.mean(p["emb"][b], axis=1)
        return jnp.sum(h @ p["w1"].T)

    found = check_model(bad, PARAMS, BATCH)
    assert sum("['emb']" in f for f in found) == 1, found
    assert sum("['w1']" in f for f in found) == 1, found


def test_an_unused_leaf_is_reported():
    """A different bug, and a quieter one: an unread parameter is still perturbed and still
    contracted, so it spends population variance on something that cannot change the fitness.
    """
    params = {**PARAMS, "dead": jnp.ones((5, 5), jnp.float32)}
    found = check_model(good, params, BATCH)
    assert len(found) == 1 and "['dead']" in found[0] and "never read" in found[0]


def test_non_matrix_leaves_are_skipped_by_default():
    """`LowRank` perturbs a non-rank-2 leaf densely, so it stays an ordinary array and how the
    model touches it is its own business. `bias` is used as a plain array by `good` and must
    not be reported."""
    assert not any("['bias']" in f for f in check_model(good, PARAMS, BATCH))
    # ...but the audit is available for a future scheme that does structure vectors.
    strict = check_model(good, PARAMS, BATCH, structured_only=False)
    assert any("['bias']" in f for f in strict), strict


def test_it_agrees_with_what_actually_happens_under_lowrank():
    """The check is only worth having if it predicts the real failure. A model it passes must
    run under `LowRank`, and one it flags must raise."""
    from shardes.strategies.lowrank import LowRank

    strategy = LowRank(r=1)
    pert = strategy.sample(jax.random.key(0), PARAMS, jnp.arange(4))

    assert check_model(good, PARAMS, BATCH) == []
    out = strategy.apply(good, PARAMS, pert, 0.1)(BATCH)
    assert out.shape == (4,)

    def bad(p, b):
        return jnp.sum(jnp.mean(embed(p["emb"], b), axis=1) @ p["w1"].T)

    assert check_model(bad, PARAMS, BATCH)
    with pytest.raises(TypeError):
        strategy.apply(bad, PARAMS, pert, 0.1)(BATCH)


# --------------------------------------------------------------------------------------
# Pytree-shaped objectives. Review finding H2, 2026-08-11.
# --------------------------------------------------------------------------------------


def test_a_pytree_walking_objective_is_reported():
    """The one misuse that is silently wrong at runtime rather than loud.

    Every other unrouted access raises under a real `LowRankWeight`: the dunders refuse `@`,
    indexing, iteration and elementwise arithmetic. A *tree walk* does not. `LowRankWeight` is
    a registered pytree of `(w, a, b, scale)`, so `jax.tree.leaves(params)` descends into the
    factors and hands back four ordinary arrays. Arithmetic on them works. The model computes
    a different function and nothing complains.

    Measured, at `n=2` with a rank-1 perturbation: a tree-L2 objective returned
    `[24.93, 30.97]` where materialising `W + sigma * (A B^T)` and taking its L2 gives
    `[16.52, 16.89]`.

    So `check_model` is the guard, and this asserts it fires.
    """
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}

    def tree_objective(p, _x):
        return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p))

    findings = check_model(tree_objective, params, jnp.ones((2, 4)))
    assert len(findings) == 1
    assert "tree.leaves" in findings[0], "the finding must name the pytree case"
    assert "computes a different function" in findings[0]


def test_a_correct_model_with_a_weight_decay_term_is_still_reported():
    """The realistic shape of the mistake, and the one a spot check would miss.

    Nobody writes a model that only walks the tree. They write one that routes its matmuls
    properly and then adds a regulariser over `jax.tree.leaves`, which is the idiom every
    other JAX codebase uses. The matmul being correct must not excuse the rest.
    """
    params = {"w": jnp.ones((4, 4)), "b": jnp.ones((4,))}

    def with_decay(p, x):
        return jnp.sum(dense(x, p["w"])) + 0.1 * sum(
            jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p)
        )

    assert check_model(with_decay, params, jnp.ones((2, 4)))
