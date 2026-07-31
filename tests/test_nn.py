"""The two seams a structured strategy reaches a model through.

`dense` covers `x @ W.T` and `embed` covers `W[ids]`. A model must be written against them
for a structured strategy to reach it, so the dispatch has to be exactly right and has to
cost nothing when unused. An embedding is a gather rather than a matmul, which is why one
seam was not enough (BACKLOG B4).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from shardes.nn import GatherableWeight, StructuredWeight, dense, embed


class Factored(NamedTuple):
    """A test double: a structured weight that happens to equal `base + a @ b.T`.

    Not LowRank, and deliberately so. The seam should work for any weight that knows how
    to apply itself, and proving that with a stand-in keeps this test from encoding one
    factorisation's conventions before that strategy is written.
    """

    base: jax.Array
    a: jax.Array
    b: jax.Array

    def apply_to(self, x):
        return x @ self.base.T + (x @ self.b) @ self.a.T


def test_array_weight_is_a_plain_matmul():
    x = jax.random.normal(jax.random.key(0), (4, 6), dtype=jnp.float32)
    w = jax.random.normal(jax.random.key(1), (5, 6), dtype=jnp.float32)
    assert jnp.allclose(dense(x, w), x @ w.T)


def test_structured_weight_is_dispatched_to():
    """The dispatch, and the equivalence it must preserve."""
    k = jax.random.split(jax.random.key(2), 4)
    x = jax.random.normal(k[0], (4, 6), dtype=jnp.float32)
    base = jax.random.normal(k[1], (5, 6), dtype=jnp.float32)
    a = jax.random.normal(k[2], (5, 2), dtype=jnp.float32)
    b = jax.random.normal(k[3], (6, 2), dtype=jnp.float32)

    w = Factored(base, a, b)
    assert isinstance(w, StructuredWeight)
    assert jnp.allclose(dense(x, w), dense(x, base + a @ b.T), atol=1e-4)


def test_a_structured_weight_is_a_pytree():
    """It travels inside the params tree through jit, vmap and shard_map, so it has to
    flatten. A frozen dataclass would not without registration."""
    w = Factored(jnp.zeros((5, 6)), jnp.zeros((5, 2)), jnp.zeros((6, 2)))
    leaves = jax.tree.leaves(w)
    assert len(leaves) == 3
    assert jax.tree.structure(w) == jax.tree.structure(
        jax.tree.map(lambda z: z, w)
    )


def test_dispatch_survives_jit_and_vmap():
    k = jax.random.split(jax.random.key(3), 4)
    x = jax.random.normal(k[0], (7, 4, 6), dtype=jnp.float32)
    w = Factored(
        jax.random.normal(k[1], (5, 6), dtype=jnp.float32),
        jax.random.normal(k[2], (5, 2), dtype=jnp.float32),
        jax.random.normal(k[3], (6, 2), dtype=jnp.float32),
    )
    want = jax.vmap(lambda xi: dense(xi, w))(x)
    assert jnp.allclose(jax.jit(lambda xx: jax.vmap(lambda xi: dense(xi, w))(xx))(x), want)


def test_a_plain_array_is_not_mistaken_for_structured():
    """jax.Array has no apply_to, but runtime_checkable Protocols only check attribute
    presence, so this is worth pinning rather than assuming."""
    assert not isinstance(jnp.zeros((3, 3)), StructuredWeight)


# --------------------------------------------------------------------------------------
# embed: the gather seam. BACKLOG B4.
# --------------------------------------------------------------------------------------


def test_embed_matches_the_materialized_table():
    """The identity the seam rests on: indexing distributes over the sum.

        (E + s A B^T)[ids] == E[ids] + s * A[ids] @ B^T

    Exact, not approximate, so this is an equality test rather than a tolerance one — the
    only slack is float reassociation in the matmul.
    """
    from shardes.strategies.lowrank import LowRankWeight

    v, d, r, b = 64, 8, 2, 5
    k = jax.random.split(jax.random.key(0), 4)
    w = LowRankWeight(jax.random.normal(k[0], (v, d), jnp.float32),
                      jax.random.normal(k[1], (v, r), jnp.float32),
                      jax.random.normal(k[2], (d, r), jnp.float32),
                      jnp.float32(0.1))
    ids = jax.random.randint(k[3], (b,), 0, v)

    want = (w.w + w.scale * (w.a @ w.b.T))[ids]
    assert jnp.allclose(embed(w, ids), want, atol=1e-5)


def test_embed_never_materializes_the_table():
    """Invariant 3 for the embedding path, traced rather than profiled.

    The banned shape is the per-member table `(n_members, V, d)`. That array is what makes
    the embedding the most expensive leaf to perturb densely, and avoiding it is the entire
    point of B4: at V = 50k the table dwarfs every other parameter in the model.
    """
    from shardes.strategies.lowrank import LowRank

    v, d, n = 128, 8, 4
    params = {"emb": jnp.zeros((v, d), jnp.float32)}
    strategy = LowRank(r=1)
    pert = strategy.sample(jax.random.key(0), params, jnp.arange(n))
    ids = jnp.array([3, 9, 40])

    def model(p, i):
        return jnp.sum(embed(p["emb"], i))

    jaxpr = jax.make_jaxpr(lambda: strategy.apply(model, params, pert, 0.1)(ids))()
    shapes = [tuple(v_.aval.shape) for eqn in jaxpr.jaxpr.eqns for v_ in eqn.outvars
              if hasattr(v_, "aval") and hasattr(v_.aval, "shape")]
    assert (n, v, d) not in shapes, f"materialized the per-member table: {sorted(set(shapes))}"


def test_embed_passes_plain_arrays_through():
    """An unstructured table is just indexed, so IIDGaussian and SeedRegenerated are
    unaffected by the seam existing."""
    table = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    ids = jnp.array([0, 3])
    assert jnp.array_equal(embed(table, ids), table[ids])


def test_embed_refuses_a_multiplyable_but_not_indexable_weight():
    """A structured weight that only implements `apply_to` must fail loudly rather than
    silently taking the array branch and indexing an object that is not a table."""
    class OnlyMultiplies:
        def apply_to(self, x):
            return x

    with pytest.raises(TypeError, match="not a GatherableWeight"):
        embed(OnlyMultiplies(), jnp.array([0, 1]))


def test_embed_and_dense_agree_on_the_same_weight():
    """One `LowRankWeight` serves both seams, and they must describe the same matrix.

    `dense(one_hot(i), W)` is row `i` of `W`, which is what `embed(W, [i])` returns. If the
    two disagree the weight means different things depending on how it is used, which is
    exactly the bug a second seam could introduce.
    """
    from shardes.strategies.lowrank import LowRankWeight

    v, d, r = 16, 6, 2
    k = jax.random.split(jax.random.key(1), 3)
    w = LowRankWeight(jax.random.normal(k[0], (v, d), jnp.float32),
                      jax.random.normal(k[1], (v, r), jnp.float32),
                      jax.random.normal(k[2], (d, r), jnp.float32),
                      jnp.float32(0.3))
    i = 5
    via_embed = embed(w, jnp.array([i]))[0]
    # dense(x, W) is x @ W.T, so a one-hot in R^d picks a *column* block; transpose the roles
    # by asking for the row directly out of the materialized equivalent instead.
    materialized = w.w + w.scale * (w.a @ w.b.T)
    assert jnp.allclose(via_embed, materialized[i], atol=1e-5)
    assert jnp.allclose(dense(jnp.eye(d, dtype=jnp.float32), w)[:, i],
                        materialized[i], atol=1e-5)
