"""Fitness shaping: fitness (n,) -> weights (n,).

Group-relative (GRPO-style) shaping is the thing that makes ES competitive with GRPO on
reasoning and is verified absent from evosax's shaping module. Small surface, high leverage
(docs/02-phase1-sharded-core.md C1.6). Not written yet.

Centered ranks need a global sort over all N fitnesses, so once this is sharded it is a
synchronization barrier: an all_gather of N scalars plus a wait, every generation. Cheap in
bytes, not free in latency. Phase 2 measures what it costs (E10).

Shaping is also discontinuous in epsilon, which is why Phase 0 sweeps with and without it.
QMC's advantage rests on bounded Hardy-Krause variation and rank transforms destroy it
(docs/00-context.md, obstacle 2).

**Only `none` and `centered` leave the estimator unbiased.** `centered_ranks` is not
estimating grad f at all: it is a deliberately different update direction, so its bias
against grad f will not go to zero and should not be read as a failure.
"""

import jax.numpy as jnp

from shardes.types import Array


def none(fitness: Array) -> Array:
    """Raw fitness. Exactly unbiased, and the noisiest of the three."""
    return fitness


def centered(fitness: Array) -> Array:
    """Mean-subtracted, with the n/(n-1) correction that keeps it unbiased.

    Subtracting the mean is the standard control variate and it is not free: `f_bar`
    contains `f_i`, which correlates with `eps_i`, so `E[f_bar eps_i] = (1/n) E[f_i eps_i]`
    and the naive version estimates `(1 - 1/n) grad f`. At n = 30, Qiu et al.'s
    population, that is a 3.3% systematic underestimate that looks like a slightly wrong
    learning rate and never surfaces as a failure.

    Measured on the quadratic at n = 16 over 40k replicates: naive mean subtraction gives
    E[g]/truth = 0.9375, exactly 1 - 1/16, while cutting the standard deviation from 4.25
    to 2.37. Take the variance and correct the scale.
    """
    n = fitness.shape[0]
    if n < 2:
        return fitness
    return (fitness - jnp.mean(fitness)) * (n / (n - 1))


def centered_ranks(fitness: Array) -> Array:
    """Rank transform to [-0.5, 0.5]. Depends only on the ordering of the fitnesses.

    The standard ES shaping, and what makes ES robust to outliers and to reward scale. It
    is **not** an estimator of grad f, so the bias check is descriptive here, not a gate.
    """
    n = fitness.shape[0]
    if n < 2:
        return jnp.zeros_like(fitness)
    ranks = jnp.argsort(jnp.argsort(fitness)).astype(fitness.dtype)
    return ranks / (n - 1) - 0.5


BY_NAME = {"none": none, "centered": centered, "centered_ranks": centered_ranks}
