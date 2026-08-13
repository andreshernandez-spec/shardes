"""Estimator-quality metrics. docs/01-phase0-estimator-harness.md C0.5.

Every function here takes pytrees and never flattens them into one vector: inner products
are summed leaf by leaf. That is not only invariant 1 hygiene. It is the only form that
survives sharding, because a concatenated vector would have to be gathered onto one device
first, which is exactly the thing this library exists to avoid.

Accumulation is f32 throughout, even for bf16 inputs. Summing 2^18 bf16 terms loses
several digits (docs/conventions.md, "Numerics").
"""

import jax
import jax.numpy as jnp

from shardes.types import Array, PyTree


def _check_same_structure(a: PyTree, b: PyTree) -> None:
    """zip over leaves truncates silently, so mismatched trees would give a plausible
    wrong number instead of an error."""
    sa, sb = jax.tree.structure(a), jax.tree.structure(b)
    if sa != sb:
        raise ValueError(f"pytree structure mismatch: {sa} vs {sb}")


def tree_vdot(a: PyTree, b: PyTree) -> Array:
    """Sum of <a_leaf, b_leaf> over leaves, accumulated in f32. Scalar."""
    _check_same_structure(a, b)
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    if not leaves_a:
        raise ValueError("empty pytree has no inner product")
    # **Shapes, not just structure.** `_check_same_structure` compares the pytree and
    # `jnp.vdot` flattens whatever it is given, so leaves of shape `(2, 3)` and `(6,)` used to
    # produce a plausible number instead of an error. An inner product between two differently
    # shaped tensors is a caller error every time, and this one is easy to reach: it is what a
    # transposed leaf or a reshaped parameter looks like.
    for x, y in zip(leaves_a, leaves_b):
        if jnp.shape(x) != jnp.shape(y):
            raise ValueError(
                f"tree_vdot got leaves of shape {jnp.shape(x)} and {jnp.shape(y)}. The "
                "structures match, so this passed the pytree check, and jnp.vdot would "
                "flatten both and return a number. There is no inner product between "
                "differently shaped tensors."
            )
    terms = [
        jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
        for x, y in zip(leaves_a, leaves_b)
    ]
    return jnp.sum(jnp.stack(terms))


def tree_norm(a: PyTree) -> Array:
    return jnp.sqrt(tree_vdot(a, a))


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    _check_same_structure(a, b)
    return jax.tree.map(lambda x, y: x - y, a, b)


def cosine_similarity(g_hat: PyTree, grad: PyTree) -> Array:
    """cos(g_hat, grad).

    The headline metric for Gate G0. Scale-free, which is the point: ES multiplies the
    estimate by a learning rate anyway, so what matters is whether the direction is
    useful, not whether the magnitude is right.
    """
    return tree_vdot(g_hat, grad) / (tree_norm(g_hat) * tree_norm(grad))


def relative_mse(g_hat: PyTree, grad: PyTree) -> Array:
    """||g_hat - grad||^2 / ||grad||^2."""
    diff = tree_sub(g_hat, grad)
    return tree_vdot(diff, diff) / tree_vdot(grad, grad)


def relative_bias(mean_g_hat: PyTree, grad: PyTree) -> Array:
    """||E[g_hat] - grad|| / ||grad||, given an average over replicates.

    Must go to zero for every scheme. If it does not for scrambled Sobol, the scrambling
    is wrong, which is the specific failure this metric exists to catch: a deterministic
    digital net is biased at fixed N and the estimate feeds SGD.
    """
    return tree_norm(tree_sub(mean_g_hat, grad)) / tree_norm(grad)
