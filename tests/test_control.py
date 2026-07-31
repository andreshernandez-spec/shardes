"""The MuJoCo Playground adapter. Phase 1 C1.7, and Gate G1 criterion 3.

Skipped entirely without the `tasks` extra, so the suite still runs on a checkout that has
only `jax` and `numpy`. Marked slow throughout: MuJoCo 3.11 runs physics through
`mujoco_warp`, which JIT-compiles its own kernels on first use and costs ~14 s before any
test body executes. That is startup, not per-step, but the fast tier should not pay it.

Episode lengths here are deliberately tiny. These pin the *contract* — that both published
algorithms drive a real environment through one API, that the policy reaches the low-rank
seam, that termination is masked — not that ES solves cartpole.
"""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("mujoco_playground", reason="needs the `tasks` extra")

from shardes import sharding, shaping  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import control  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

pytestmark = pytest.mark.slow

ENV = "CartpoleBalance"
STEPS = 20


@pytest.fixture(scope="module")
def task():
    """One env for the module. `registry.load` plus kernel compilation is the whole cost."""
    return control.make(ENV, jax.random.key(0), hidden=(8,), episode_length=STEPS)


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(lambda: Mirrored(LowRank(r=1)), id="eggroll"),
        pytest.param(lambda: Mirrored(SeedRegenerated()), id="qiu"),
    ],
)
def test_both_published_algorithms_drive_a_real_environment(task, strategy):
    """Gate G1 criterion 3. One API, one argument different, a real physics environment.

    The generation is jitted as a whole rather than stepped eagerly, and that is required
    rather than stylistic: MJX commits all 124 of its model arrays to device 0 while the
    perturbation is committed across the mesh, and mixing two committed placements raises
    "Received incompatible devices" from inside MuJoCo's FFI. One `jit` resolves placement
    at trace time. See control.py's docstring.
    """
    params, model = task
    keys = control.episode_keys(jax.random.key(1), 2)
    es = ShardedES(strategy=strategy(), n=8, sigma=0.02, lr=0.05,
                   mesh=sharding.make_mesh(4), shaping=shaping.group_relative)
    state = es.init(jax.random.key(2), params)

    @jax.jit
    def generation(state, keys):
        pert, state = es.ask(state)
        returns = es.apply(model, state, pert)(keys)
        return es.tell(state, pert, -returns), returns

    state, returns = generation(state, keys)
    assert returns.shape == (8, 2)
    assert bool(jnp.all(jnp.isfinite(returns)))
    assert jax.tree.structure(state.params) == jax.tree.structure(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(state.params))


def test_the_policy_reaches_the_low_rank_seam(task):
    """`LowRank` substitutes a `LowRankWeight` into the params tree, so a policy written as
    `obs @ W.T` would be handed one and raise. This is the constraint a user porting a Flax
    policy hits first, so it gets a test rather than only a docstring.
    """
    from shardes.nn import StructuredWeight
    from shardes.strategies.lowrank import LowRankWeight

    params, _ = task
    w = params[0]["w"]
    structured = LowRankWeight(w, jnp.zeros((w.shape[0], 1)), jnp.zeros((w.shape[1], 1)),
                               jnp.float32(0.1))
    assert isinstance(structured, StructuredWeight)
    # The policy must accept it where a plain array would go.
    swapped = [{"w": structured, "b": params[0]["b"]}, *params[1:]]
    out = control.policy(swapped, jnp.zeros((w.shape[1],), jnp.float32))
    assert out.shape == (params[-1]["w"].shape[0],) and bool(jnp.all(jnp.isfinite(out)))


def test_termination_is_masked_not_ignored():
    """A terminated environment keeps emitting rewards, and summing them scores a policy on
    time it spent dead.

    Against a stub that actually terminates, and against the *real* `control.rollout` rather
    than a copy of it: CartpoleBalance never terminates early, so a test written against it
    would assert nothing at all.
    """
    from typing import NamedTuple

    class S(NamedTuple):     # a NamedTuple, so lax.scan can carry it
        obs: jnp.ndarray
        reward: jnp.ndarray
        done: jnp.ndarray
        step: jnp.ndarray

    class Stub:
        observation_size, action_size = 2, 1

        def reset(self, key):
            return S(jnp.zeros((2,)), jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0))

        def step(self, s, a):
            nxt = s.step + 1
            # dies on step 3, then keeps paying out
            return S(jnp.zeros((2,)), jnp.float32(1.0), (nxt >= 3).astype(jnp.float32), nxt)

    flat = [{"w": jnp.zeros((1, 2)), "b": jnp.zeros((1,))}]
    got = float(control.rollout(Stub(), flat, jax.random.key(0), 10))
    assert got == 3.0, f"rewards after termination were counted: {got}"
    # and the same when the scan stops exactly at the terminating step
    assert float(control.rollout(Stub(), flat, jax.random.key(0), 3)) == 3.0


def test_episode_keys_are_shared_across_members(task):
    """Common random numbers. Every member sees the same initial states, so the fitness
    spread reflects the perturbations and not who drew an easy start."""
    keys = control.episode_keys(jax.random.key(3), 4)
    assert keys.shape == (4,)
    # Same key in, same keys out: the batch is a pure function of the generation key, and
    # is not derived per member anywhere.
    assert jnp.array_equal(jax.random.key_data(keys),
                           jax.random.key_data(control.episode_keys(jax.random.key(3), 4)))


def test_a_one_dimensional_shaping_on_multi_episode_fitness_is_refused(task):
    """The C1.6 contract widening left a footgun: `centered_ranks` given (n, episodes) ranks
    along the last axis and returns (n, episodes) without complaint, and the failure then
    surfaces as an einsum subscript error inside `contract`. `tell` catches it instead."""
    params, model = task
    keys = control.episode_keys(jax.random.key(1), 2)
    es = ShardedES(strategy=Mirrored(LowRank(r=1)), n=8, sigma=0.02, lr=0.05,
                   mesh=sharding.make_mesh(4))  # default shaping is centered_ranks
    state = es.init(jax.random.key(2), params)

    @jax.jit
    def generation(state, keys):
        pert, state = es.ask(state)
        returns = es.apply(model, state, pert)(keys)
        return es.tell(state, pert, -returns)

    # Raised during tracing, so it propagates out of the jit. Written as a jitted generation
    # rather than eagerly because the eager path hits the device-placement trap this module
    # documents, and a test that failed for *that* reason would be testing the wrong thing.
    with pytest.raises(ValueError, match="shaping returned"):
        generation(state, keys)
