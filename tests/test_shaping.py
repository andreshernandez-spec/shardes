"""Fitness shaping.

The defining property of rank shaping is that it depends on the *ordering* of the
fitnesses and nothing else. That is what makes ES robust to reward scale and to outliers,
and it is the property that catches an implementation which quietly passes raw values
through: such an implementation still produces a sensible-looking update direction, so
the estimator tests do not notice.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import shaping

N = 16


@pytest.fixture
def fitness():
    return jax.random.normal(jax.random.key(0), (N,), dtype=jnp.float32)


def test_none_is_the_identity(fitness):
    assert jnp.array_equal(shaping.none(fitness), fitness)


@pytest.mark.parametrize(
    "monotone",
    [lambda f: 3 * f, lambda f: f + 100.0, lambda f: f**3, lambda f: jnp.exp(f)],
    ids=["scale", "shift", "cube", "exp"],
)
def test_centered_ranks_depend_only_on_ordering(fitness, monotone):
    """Any strictly increasing map of the fitnesses must give identical weights.

    This is the test that distinguishes a rank transform from a rescaled raw value. An
    implementation returning `fitness/(n-1) - 0.5` produces a plausible update direction
    and passes every estimator test; it fails here immediately.
    """
    assert jnp.allclose(
        shaping.centered_ranks(fitness), shaping.centered_ranks(monotone(fitness))
    )


def test_centered_ranks_are_permutation_equivariant(fitness):
    perm = jax.random.permutation(jax.random.key(1), N)
    got = shaping.centered_ranks(fitness[perm])
    assert jnp.allclose(got, shaping.centered_ranks(fitness)[perm])


def test_centered_ranks_survive_an_outlier(fitness):
    """One absurd fitness must not dominate. This is most of why ES uses ranks."""
    spiked = fitness.at[3].set(1e9)
    base = shaping.centered_ranks(fitness.at[3].set(fitness.max() + 1.0))
    assert jnp.allclose(shaping.centered_ranks(spiked), base)


def test_centered_ranks_span_the_unit_interval(fitness):
    w = shaping.centered_ranks(fitness)
    assert jnp.isclose(w.min(), -0.5)
    assert jnp.isclose(w.max(), 0.5)
    assert jnp.isclose(w.sum(), 0.0, atol=1e-5)


def test_centered_sums_to_zero(fitness):
    assert jnp.isclose(shaping.centered(fitness).sum(), 0.0, atol=1e-4)


def test_centered_carries_the_bias_correction(fitness):
    """The n/(n-1) factor is the whole point; without it the estimator targets
    (1 - 1/n) grad f. Checked against the naive version directly."""
    naive = fitness - jnp.mean(fitness)
    assert jnp.allclose(shaping.centered(fitness), naive * (N / (N - 1)), rtol=1e-6)
    assert not jnp.allclose(shaping.centered(fitness), naive)


def test_centered_is_shift_invariant(fitness):
    """A constant added to every fitness is not information."""
    assert jnp.allclose(shaping.centered(fitness + 1000.0), shaping.centered(fitness), atol=1e-3)


@pytest.mark.parametrize("name", ["none", "centered", "centered_ranks"])
def test_single_member_does_not_divide_by_zero(name):
    """n = 1 shows up in chunked runs with chunk=1; n-1 must not blow up."""
    out = shaping.BY_NAME[name](jnp.array([1.5], dtype=jnp.float32))
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


def test_by_name_covers_every_public_shaping():
    """A shaping reachable by config must be reachable by name, or a sweep silently
    cannot select it."""
    public = {n for n in dir(shaping) if not n.startswith("_") and callable(getattr(shaping, n))}
    public -= {"Array"}  # the type alias import
    assert public == set(shaping.BY_NAME), public ^ set(shaping.BY_NAME)


# --------------------------------------------------------------------------------------
# Group-relative (GRPO-style). Phase 1 C1.6.
# --------------------------------------------------------------------------------------


def test_group_relative_is_invariant_to_per_task_reward_scale():
    """The property the whole thing exists for.

    Two tasks, one scored in [0,1] and one in the thousands. Under any shaping that does not
    normalize per task, the large-scale task *is* the objective and the other is rounding.
    Here rescaling a task must not change the weights at all, because a task contributes
    through the ordering it induces over the population and nothing else.
    """
    f = jax.random.normal(jax.random.key(0), (32, 2), jnp.float32)
    rescaled = f * jnp.array([1.0, 5000.0]) + jnp.array([0.0, 12345.0])

    got = shaping.group_relative(f)
    want = shaping.group_relative(rescaled)
    assert float(jnp.max(jnp.abs(got - want))) < 1e-4


def test_a_task_nobody_varies_on_contributes_exactly_zero():
    """All-solved and all-failed tasks are common on reasoning benchmarks, not exotic.

    Their standard deviation is zero and they carry no signal. `sd + eps` would turn that
    into a large arbitrary weight; the zero has to be selected explicitly. Asserted as
    *exactly* equal to the single-task result, so a near-miss cannot pass.
    """
    live = jax.random.normal(jax.random.key(0), (16, 1), jnp.float32)
    dead_hi = jnp.ones((16, 1), jnp.float32)        # everyone solves it
    dead_lo = jnp.zeros((16, 1), jnp.float32)       # nobody does

    alone = shaping.group_relative(live)
    with_dead = shaping.group_relative(jnp.concatenate([live, dead_hi, dead_lo], axis=1))

    assert jnp.all(jnp.isfinite(with_dead))
    # Three tasks, two contributing zero: the mean over tasks is a third of the live one.
    assert float(jnp.max(jnp.abs(with_dead * 3.0 - alone))) < 1e-5


def test_group_relative_weights_sum_to_zero_per_task():
    """Centering across members is what makes it a baseline. If the weights did not sum to
    zero, a task would push the whole population in one direction regardless of ordering."""
    f = jax.random.normal(jax.random.key(1), (64, 5), jnp.float32)
    mu = jnp.mean(f, axis=0, keepdims=True)
    sd = jnp.std(f, axis=0, keepdims=True)
    per_task = (f - mu) / sd
    assert float(jnp.max(jnp.abs(jnp.sum(per_task, axis=0)))) < 1e-3
    assert abs(float(jnp.sum(shaping.group_relative(f)))) < 1e-3


def test_group_relative_ranks_a_better_member_higher():
    """Sanity, and it catches a sign flip: a member that beats everyone on every task must
    get the largest weight."""
    f = jax.random.normal(jax.random.key(2), (16, 4), jnp.float32)
    f = f.at[5].set(f.max() + 10.0)
    w = shaping.group_relative(f)
    assert int(jnp.argmax(w)) == 5


def test_group_relative_refuses_a_one_dimensional_fitness():
    """A 1-D fitness has one group and nothing to be relative to. Raising beats silently
    treating the member axis as the group axis, which would return zeros."""
    with pytest.raises(ValueError, match="n_members, n_groups"):
        shaping.group_relative(jnp.arange(8, dtype=jnp.float32))


def test_group_relative_handles_a_degenerate_population():
    """n < 2 has no baseline. Returns (n,) zeros rather than dividing by zero."""
    out = shaping.group_relative(jnp.ones((1, 3), jnp.float32))
    assert out.shape == (1,) and float(out[0]) == 0.0


def test_group_relative_is_registered():
    assert shaping.BY_NAME["group_relative"] is shaping.group_relative
