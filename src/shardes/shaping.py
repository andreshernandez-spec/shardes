"""Fitness shaping. Leading axis is members; every shaping returns `(n,)` weights.

Most take `(n,)`. `group_relative` takes `(n, g)` — n members scored on g tasks — which is
why the contract is stated as "leading axis is members" rather than "takes (n,)". A shaping
is free to consume trailing axes; it must not change the leading one.

Group-relative (GRPO-style) shaping is the thing that makes ES competitive with GRPO on
reasoning and is verified absent from evosax's shaping module. Small surface, high leverage
(docs/02-phase1-sharded-core.md C1.6).

**Every shaping here is a synchronization barrier once sharded**, and it is worth being
precise about why, because the reasons differ. `centered_ranks` needs a global sort over all
N fitnesses. `centered` needs the global mean. `group_relative` needs a per-task mean and
standard deviation *across members*, so it reduces over the sharded axis too. All three are
an all-gather of N scalars (or N*g) plus a wait, every generation. Cheap in bytes, not free
in latency. Phase 2 measures what it costs (E10). `none` is the only one that is not a
barrier, which is one reason to keep it.

Shaping is also discontinuous in epsilon, which is why Phase 0 sweeps with and without it.
QMC's advantage rests on bounded Hardy-Krause variation and rank transforms destroy it
(docs/00-context.md, obstacle 2).

**Only `none` and `centered` leave the estimator unbiased.** `centered_ranks` and
`group_relative` are not estimating grad f at all: they are deliberately different update
directions, so their bias against grad f will not go to zero and should not be read as a
failure.
"""

import jax
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

    **Do not compose this with Mirrored.** The pair contributes
    `((f_2k - f_bar) - (f_2k+1 - f_bar)) eps_k`, so `f_bar` cancels outright and the
    antithetic estimator is already unbiased. The n/(n-1) factor then over-corrects,
    targeting `n/(n-1) grad f`: 6.7% the wrong way at n = 16, worse as n shrinks. The
    correction belongs to the estimator-and-shaping pair, not to shaping alone.

    With Mirrored use `none`, which is already centred by construction, or
    `centered_ranks`, which is not estimating grad f in the first place.
    """
    n = fitness.shape[0]
    if n < 2:
        return fitness
    return (fitness - jnp.mean(fitness)) * (n / (n - 1))


def centered_ranks(fitness: Array) -> Array:
    """Rank transform to [-0.5, 0.5]. Depends only on the ordering of the fitnesses.

    The standard ES shaping, and what makes ES robust to outliers and to reward scale. It
    is **not** an estimator of grad f, so the bias check is descriptive here, not a gate.

    **Tied members share a rank.** `argsort(argsort(f))` gives equal fitnesses *different*
    ranks, ordered by index, so two members that score identically get weights up to a third
    of the range apart and which one is favoured is decided by the sort rather than by the
    objective. Nobody chose that; it is what the two-argsort trick does.

    It matters here more than it would elsewhere. `experiments/phase2/noisefloor.py` measures
    5 to 9 exactly tied pairs at `d=512, N=1024`, because a perturbation at `sigma=0.01`
    moves the loss by less than one float32 ulp and members collide. A pair that ties on one
    backend and differs by an ulp on another then orders either way, and one exchange near
    the middle of the population is worth about 1e-2 in the update
    (`docs/postmortem-gpu-determinism.md`). Midranks make the weights a function of the
    fitness multiset instead of the sort's tie-break.

    With no ties this is exactly `argsort(argsort(f))/(n-1) - 0.5`, bitwise. Ties are the
    only case that changes, and there the old answer was arbitrary.
    """
    n = fitness.shape[0]
    if n < 2:
        return jnp.zeros_like(fitness)
    return _midranks(fitness) / (n - 1) - 0.5


def _midranks(fitness: Array) -> Array:
    """Average rank within each group of equal values, along the last axis.

    **Shape-agnostic on purpose, like the `argsort(argsort(x))` it replaces.** The contract
    at the top of this module is `(n,)` in, `(n,)` out, and an `(n, episodes)` fitness is a
    caller error. But it is `tell` that reports that error, in the one message that names the
    fix, so this has to hand the wrong shape *through* rather than die on it. A 1-D-only
    implementation turns a clear ValueError into a TypeError from inside `concatenate`,
    which `tests/test_control.py` catches.
    """
    m = fitness.shape[-1]
    rows = fitness.reshape(-1, m)

    def one(row: Array) -> Array:
        # **float32 regardless of the fitness dtype.** Rank positions run to m, and bf16 has
        # 8 mantissa bits, so above 256 members consecutive positions stop being distinct and
        # the shaping silently compresses the population it was meant to spread out. The
        # fitness dtype is the caller's business; the ranks are this function's.
        row = row.astype(jnp.float32)
        # **Non-finite fitness is the worst outcome, and all of it ties.** `NaN != NaN`, so
        # every failed member used to start its own tie group and land on a different rank;
        # measured, `centered_ranks([0, nan, nan, 1])` gave the two NaNs 0.167 and 0.5, the
        # largest weight in the population. A diverged episode was getting the strongest
        # vote. `+inf` is the worst loss under `tell`'s descent convention, and mapping every
        # NaN onto the same value makes them tie with each other by construction.
        row = jnp.where(jnp.isnan(row), jnp.inf, row)
        order = jnp.argsort(row)
        ordered = row[order]
        # A tie group starts wherever the sorted value changes.
        starts = jnp.concatenate([jnp.array([True]), ordered[1:] != ordered[:-1]])
        group = jnp.cumsum(starts) - 1
        positions = jnp.arange(m, dtype=jnp.float32)

        # `first + (count - 1) / 2`, not `sum(positions) / count`. They agree in exact
        # arithmetic and not in float32: the sum reaches m^2/2, which passes 2^24 and stops
        # being representable. With every member tied that is 0.5 ranks of error at m = 2^16
        # and 1.4 at m = 2^18, the population EGGROLL runs. Every intermediate here stays
        # below m.
        first = jax.ops.segment_min(positions, group, num_segments=m)
        count = jax.ops.segment_sum(jnp.ones_like(positions), group, num_segments=m)
        return jnp.zeros_like(row).at[order].set((first + (count - 1) / 2)[group])

    return jax.vmap(one)(rows).reshape(fitness.shape)


def group_relative(fitness: Array) -> Array:
    """GRPO-style shaping. `fitness` is `(n, g)`: n members, g tasks. Returns `(n,)`.

    Each task is normalized **across members** before the tasks are averaged, so a task
    contributes to the update through the *ordering it induces over the population*, not
    through its reward scale. Without that, one task with rewards in the thousands and
    another with rewards in [0, 1] are not two tasks: they are one task and a rounding error.

    This is the piece that makes ES competitive with GRPO on reasoning, and it is verified
    absent from evosax's shaping module (docs/02 C1.6). It is a small surface for what it
    buys: the whole idea is that a group baseline replaces a learned value function, and ES
    already has the group.

    **A task nobody varies on contributes exactly zero, not a NaN.** If every member scores
    identically on a task — all solve it, or none do — its standard deviation is zero and
    there is no signal in it. That case is common rather than exotic on reasoning benchmarks,
    where a fraction of problems are saturated in both directions from the first generation.
    Dividing by `sd + eps` would instead turn a dead task into a large arbitrary weight, so
    the zero is selected explicitly.

    **Not an estimator of grad f**, for the same reason `centered_ranks` is not: dividing by
    a per-task standard deviation is a data-dependent rescaling, so the bias check in
    docs/01 C0.5 is descriptive here rather than a gate. The mean subtraction also carries
    the `(1 - 1/n)` shrinkage measured in Phase 0, and it is *not* corrected here — the
    correction would be a scale factor on something already rescaled by an estimated
    standard deviation, which is arithmetic without meaning. Read this as a different update
    direction, not as a noisier `centered`.

    Known sharp edge, inherited from GRPO rather than introduced here, and worth stating
    precisely because the loose version is misleading. Standardising is **scale-invariant**,
    so the weights within a group are bounded by `sqrt(n-1)` however small the spread —
    measured at ~1.5 for spreads from 1e-2 down to 1e-7 at n=8, and nothing blows up. What it
    does is give a group where the population barely differs the *same say* as one where it
    differs a lot, so a task carrying almost no signal is weighted like a task carrying a
    great deal. That is a statement about weighting **between** groups, not about magnitudes
    within one.

    The fix in the literature is to drop the `sd` divisor and keep only the centering. That is
    one line away and deliberately not the default, because the default should be the thing
    the papers ran.
    """
    if fitness.ndim != 2:
        raise ValueError(
            f"group_relative needs (n_members, n_groups), got shape {fitness.shape}. A 1-D "
            "fitness has one group and no group structure to be relative to; use `centered` "
            "or `centered_ranks` for that."
        )
    n = fitness.shape[0]
    if n < 2:
        return jnp.zeros(fitness.shape[:1], fitness.dtype)

    mu = jnp.mean(fitness, axis=0, keepdims=True)
    sd = jnp.std(fitness, axis=0, keepdims=True)
    advantage = jnp.where(sd > 0, (fitness - mu) / jnp.where(sd > 0, sd, 1.0), 0.0)
    return jnp.mean(advantage, axis=1)


#: Shaping by name. Every entry takes a leading member axis and returns `(n,)` weights;
#: `group_relative` is the one that consumes a second axis, which is why the contract is
#: stated as "leading axis is members" rather than "takes (n,)".
BY_NAME = {
    "none": none,
    "centered": centered,
    "centered_ranks": centered_ranks,
    "group_relative": group_relative,
}
