"""Mesh, NamedSharding, PartitionSpec, and the seed contract.

    from jax import shard_map      # NOT jax.experimental.shard_map, deprecated in JAX 0.8.0

Layout (docs/02-phase1-sharded-core.md C1.2):
  params        replicated across the "pop" axis, every device holds the full model
  perturbations sharded on the member axis, P("pop", None, None)
  fitnesses     sharded, P("pop")
  state         see docs/02 C1.4, the decision is open

Parameters are never sharded. ES's advantage is that every device holds the model and runs
inference independently; sharding parameters reintroduces the communication ES avoids.

**The seed contract, and why this module enforces it structurally rather than by
discipline.**

    member i's perturbation derives from jax.random.fold_in(base_key, i), where i is the
    global member index. Never the device index, never a per-device counter, never
    sequential consumption of a key stream.

`member_ids` returns `arange(n)` sharded over the member axis. Sharding partitions an array
without changing its values, so device `k` holds the literal global indices
`[k*n/D, (k+1)*n/D)` and there is nowhere for a device index to enter: the only integer a
strategy ever sees is already global. Device-count invariance falls out of the data layout
rather than being a property anyone has to maintain.

That is why `sample(base_key, params, member_ids)` takes ids rather than a count. An API
shaped `sample(key, params, n_local)` could not express this without passing a device index
alongside, which is precisely the bug the contract exists to prevent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from shardes.types import Array, PyTree

#: The one mesh axis. Members and their rollouts split along it; nothing else does.
POP = "pop"


def make_mesh(n_devices: int | None = None) -> Mesh:
    """A 1-D mesh over the member axis.

    `n_devices=None` takes every visible device. Tests drive 1, 2, 4 and 8 through the same
    code path, which is what `XLA_FLAGS=--xla_force_host_platform_device_count=8` buys:
    PartitionSpec errors, shard_map signatures and collective placement all reproduce on
    CPU. Simulated devices model bytes faithfully and interconnect not at all, so they
    settle correctness and never timing (docs/02, traps).
    """
    devices = jax.devices()
    if n_devices is None:
        n_devices = len(devices)
    if not 1 <= n_devices <= len(devices):
        raise ValueError(f"asked for {n_devices} devices, {len(devices)} visible")
    return jax.make_mesh((n_devices,), (POP,), devices=devices[:n_devices])


def n_devices(mesh: Mesh) -> int:
    return mesh.shape[POP]


def check_population(n: int, mesh: Mesh, *, paired: bool = False) -> None:
    """Raise unless `n` members shard evenly, loudly and before anything expensive runs.

    `paired=True` additionally requires an even count per device. Mirrored pairs members as
    (2k, 2k+1), and a split landing between them loses the antithetic cancellation
    *silently*: the run still produces an update, just a worse one. With both `n` and the
    device count powers of two this always holds, which is exactly why the one configuration
    where it does not would go unnoticed.
    """
    d = n_devices(mesh)
    if n % d:
        raise ValueError(
            f"population {n} does not divide across {d} devices. shard_map needs an even "
            "split, and an uneven one would change the update rather than fail."
        )
    if paired and (n // d) % 2:
        raise ValueError(
            f"population {n} over {d} devices gives {n // d} members per device, which is "
            "odd. Mirrored pairs members as (2k, 2k+1), so an odd shard splits a pair "
            "across devices and loses the antithetic cancellation without erroring."
        )


def member_ids(n: int, mesh: Mesh) -> Array:
    """(n,) int32 of GLOBAL member indices, sharded over the member axis.

    The values are the contract. Device `k` receives `[k*n/D, (k+1)*n/D)` because sharding
    partitions without renumbering, so a strategy running inside `shard_map` sees global ids
    and cannot see anything else. See the module docstring.
    """
    check_population(n, mesh)
    return jax.device_put(jnp.arange(n, dtype=jnp.int32), members(mesh))


# --------------------------------------------------------------------------------------
# Shardings. Named rather than inlined: a P() in the wrong place is a silent correctness
# bug, and these four are all of them.
# --------------------------------------------------------------------------------------


def replicated(mesh: Mesh) -> NamedSharding:
    """Params, and anything else every device needs whole."""
    return NamedSharding(mesh, P())


def members(mesh: Mesh) -> NamedSharding:
    """Fitness and member ids: one scalar per member, split on the member axis."""
    return NamedSharding(mesh, P(POP))


def per_member(mesh: Mesh, rank: int) -> NamedSharding:
    """A materialized per-member array: leading member axis, everything else whole.

    `rank` counts trailing dimensions, so an (n, m, k) perturbation leaf is `rank=2`.
    """
    return NamedSharding(mesh, P(POP, *([None] * rank)))


def shard_perturbation(pert: PyTree, mesh: Mesh, n: int) -> PyTree:
    """Place a materialized perturbation's leaves on the member axis.

    `n` is the population, and it is required rather than inferred. Leaves carrying no member
    axis have to be replicated instead — `SeedRegenerated` holds `like`, a reference to the
    unbatched params — and the only way to tell them apart is whether the leading axis is
    `n`. A divisibility test looks like it works and silently degrades to "shard everything"
    at one device, where `x.shape[0] % 1` is zero for every array.

    `Mirrored` is why this takes `n` rather than reading it off the largest leaf: its inner
    perturbation has a leading axis of `n/2`, which is a legitimate member axis at half
    resolution. Callers pass the count the leaves were built with.

    Deciding by shape rather than by strategy keeps this from having to know which strategy
    produced the tree, which is the same reason `contract` never asks.
    """
    check_population(n, mesh)

    def place(x):
        if jnp.ndim(x) and x.shape[0] == n:
            return jax.device_put(x, per_member(mesh, jnp.ndim(x) - 1))
        return jax.device_put(x, replicated(mesh))

    return jax.tree.map(place, pert)
