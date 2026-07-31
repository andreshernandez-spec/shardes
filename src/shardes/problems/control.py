"""MuJoCo Playground as an ES objective. Phase 1 C1.7.

    params, model = control.make("CartpoleBalance", key, hidden=(32, 32))
    es = ShardedES(strategy=Mirrored(LowRank(r=1)), n=1024, sigma=0.02, lr=0.05, mesh=mesh,
                   shaping=shaping.group_relative)           # (n, episodes) -> (n,)
    state = es.init(key, params)

    @jax.jit                                   # jit the whole generation; see below
    def generation(state, keys):
        pert, state = es.ask(state)
        returns = es.apply(model, state, pert)(keys)         # (n, episodes)
        return es.tell(state, pert, -returns)                # note the sign

`model` returns one value per episode, so the fitness is `(n, episodes)` and the shaping has
to reduce that axis: `group_relative` does, the 1-D shapings do not and `tell` will say so.
Averaging the episodes yourself and using `centered_ranks` is equally valid and throws away
the per-episode structure that `group_relative` exists to use.

**Jit the whole generation rather than calling ask/apply/tell eagerly.** MJX commits all 124
of its model arrays to device 0 (checked, not assumed), while the perturbation is committed
across the mesh, and mixing two committed placements raises "Received incompatible devices"
from somewhere deep inside MuJoCo's FFI. Under one `jit`, JAX resolves placement at trace
time and the conflict does not arise. This is not a MuJoCo quirk to work around so much as
the way the core is meant to be driven, but the failure it produces is unreadable, so it is
worth stating.

Needs the `tasks` extra (`pip install -e ".[tasks]"`). Import is lazy so the library floor
stays `jax` + `numpy`; nothing under `shardes/` imports this module.

**The sign.** `tell` descends on what it is given (`core.py`), so a *reward* is negated
before it goes in. This module returns episode return, which is the thing to maximize, and
does not negate it for you: a helper that silently flipped a sign would be worse than the
one line at the call site.

---

**The policy goes through `shardes.nn.dense`, and that is load-bearing rather than
stylistic.** `LowRank` perturbs by substituting a `LowRankWeight` into the params tree; a
policy written as `obs @ W.T` would be handed one of those and raise. Writing it against
`dense` is the whole reason a low-rank strategy needs no jaxpr interpreter (docs/01 C0.1),
and it is the constraint a user porting a Flax policy will hit first.

**Deterministic policy, on purpose.** ES perturbs *parameters* and evaluates the resulting
deterministic behaviour; it does not sample actions. That is the difference from policy
gradient, and it is why the whole rollout is a pure function of `(params, key)` with the key
used only for the environment's own reset randomization.

**Termination is masked, not ignored.** `lax.scan` needs a static length, so every rollout
runs `episode_length` steps whatever happens. A terminated environment keeps emitting
rewards, and summing them would score a policy on time it spent dead. The carry tracks an
`alive` flag and zeroes reward after the first `done`, which is the difference between
"balanced the pole for 200 steps" and "fell over at step 12 and kept collecting".
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from shardes.nn import dense
from shardes.types import Array, Key, PyTree


def _load(env_name: str):
    """Import Playground lazily, and say what to install if it is missing."""
    try:
        from mujoco_playground import registry
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            "control tasks need MuJoCo Playground: pip install -e '.[tasks]'. It is free "
            "and Apache 2.0; the paid MuJoCo licence ended when DeepMind acquired it in "
            "2021 (docs/02-phase1-sharded-core.md C1.7)."
        ) from exc
    return registry.load(env_name)


def init_policy(
    key: Key, obs_size: int, action_size: int, hidden: Sequence[int] = (32, 32)
) -> PyTree:
    """A small tanh MLP, as a pytree of `(w, b)` pairs.

    Scaled by `1/sqrt(fan_in)` so the initial policy is near-constant rather than saturated:
    a tanh policy that starts pinned at +/-1 has no gradient signal for ES to find either,
    since every perturbation produces the same action.
    """
    sizes = [obs_size, *hidden, action_size]
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        {
            "w": jax.random.normal(k, (m, n), jnp.float32) / jnp.sqrt(jnp.float32(n)),
            "b": jnp.zeros((m,), jnp.float32),
        }
        for k, n, m in zip(keys, sizes[:-1], sizes[1:])
    ]


def policy(params: PyTree, obs: Array) -> Array:
    """Tanh MLP. Every matmul goes through `dense` so structured weights substitute."""
    h = obs
    for layer in params[:-1]:
        h = jnp.tanh(dense(h, layer["w"]) + layer["b"])
    return jnp.tanh(dense(h, params[-1]["w"]) + params[-1]["b"])


def make(
    env_name: str,
    key: Key,
    *,
    hidden: Sequence[int] = (32, 32),
    episode_length: int = 200,
) -> tuple[PyTree, Callable[[PyTree, Array], Array]]:
    """`(init_params, model)` for a Playground env.

    `model(params, keys)` returns the summed reward per episode. `keys` is a batch of
    typed PRNG keys, one per episode, so the returned shape is `keys.shape`. Several
    episodes per member is how the fitness stops being dominated by which initial state a
    member happened to draw — and it is exactly the `(n, g)` layout `shaping.group_relative`
    consumes, with the episodes as groups.

    `episode_length` is static because `lax.scan` needs it to be.
    """
    env = _load(env_name)
    params = init_policy(key, env.observation_size, env.action_size, hidden)

    def model(params: PyTree, episode_keys: Array) -> Array:
        return jax.vmap(rollout, in_axes=(None, None, 0, None))(
            env, params, episode_keys, episode_length
        )

    return params, model


def rollout(env, params: PyTree, episode_key: Key, episode_length: int) -> Array:
    """Summed reward over one episode, with reward masked after termination.

    Module level and taking `env` explicitly so the masking can be tested against a stub
    environment that actually terminates. CartpoleBalance never does, so a test written
    against it would assert nothing, and a test that reimplements this loop would assert
    that the copy is right.
    """
    state = env.reset(episode_key)

    def step(carry, _):
        state, alive = carry
        state = env.step(state, policy(params, state.obs))
        # Reward is credited only while the episode is still running. `alive` drops on the
        # step that reports done and never comes back.
        reward = state.reward * alive
        return (state, alive * (1.0 - state.done)), reward

    # ones_like(state.reward), not jnp.float32(1.0): a bare literal is committed to device 0
    # while params may be committed across the mesh, and a scan carry holding both raises
    # "incompatible devices". Deriving the flag from an array that came out of `reset`
    # inherits its placement. Same shape of trap as the accumulator in
    # SeedRegenerated.contract, and it appears wherever a constant joins a carry.
    alive = jnp.ones_like(state.reward)
    _, rewards = jax.lax.scan(step, (state, alive), None, length=episode_length)
    return jnp.sum(rewards)


def episode_keys(key: Key, episodes: int) -> Array:
    """`episodes` env-reset keys.

    Deliberately *not* derived per member. Every member of a generation is evaluated on the
    **same** initial states, which is common random numbers: the fitness differences then
    reflect the perturbations rather than which member drew an easy start. Without it the
    estimator is measuring the environment's variance, and at the population sizes this
    library targets that noise does not average away, it just costs members.
    """
    return jax.random.split(key, episodes)
