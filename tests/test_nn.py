"""The linear seam.

`dense` is the one thing a model must be written against for a structured strategy to
reach it, so the dispatch has to be exactly right and has to cost nothing when unused.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shardes.nn import StructuredWeight, dense


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
