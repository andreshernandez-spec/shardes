"""Mesh, NamedSharding, PartitionSpec, and the seed contract.

    from jax import shard_map      # NOT jax.experimental.shard_map, deprecated in JAX 0.8.0

Layout (docs/02-phase1-sharded-core.md C1.2):
  params        replicated across the "pop" axis, every device holds the full model
  perturbations sharded on the member axis, P("pop")
  fitnesses     sharded, P("pop")
  state         replicated, not sharded (docs/02 C1.4, settled 2026-07-31)

The perturbation layout is not placed by hand. `ShardedES.apply` constrains what it
*produces* to the member axis, and GSPMD propagates that backwards to shard the perturbation
behind it. Placing the perturbation directly does not distribute the evaluation, which is
why the helper that used to do it is gone (`docs/diagnosis-replicated-evaluation.md`).

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
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from shardes.types import Array

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
    # AxisType.Auto, not jax.make_mesh's Explicit default. This is a load-bearing choice,
    # not a stylistic one: see AXIS_TYPE_NOTE below.
    return jax.make_mesh((n_devices,), (POP,), axis_types=(AxisType.Auto,),
                         devices=devices[:n_devices])


AXIS_TYPE_NOTE = """Why the mesh is Auto rather than Explicit.

JAX 0.11's `jax.make_mesh` defaults to `AxisType.Explicit`, which carries sharding in the
type system and refuses operations whose output sharding is ambiguous. Under Explicit:

  - `with_sharding_constraint(x, replicated(mesh))` is silently *ignored*: the result still
    reports P("pop"). Measured, not assumed.
  - `contract`'s `einsum("n,n...->...")` then raises ShardingTypeError, because both
    operands are sharded along the contracted axis and JAX will not guess where the output
    goes. Fixing that means passing `out_sharding=` at the einsum.

That last point is the whole argument. The einsum lives inside `IIDGaussian.contract`, so
Explicit mode would push a mesh-aware annotation down into every strategy. The strategies
were written in Phase 0 knowing nothing about devices, and the protocol
(`sample`/`apply`/`contract`) is deliberately silent about them: that is what let coupling
become a constructor argument and what keeps a user-defined strategy first-class. Trading
that for compile-time sharding checks is a bad trade.

Auto uses classic GSPMD propagation, `with_sharding_constraint` behaves as written, and the
strategies stay sharding-agnostic. `shard_map` runs its body under `AxisType.Manual`
regardless, so Strategy B is unaffected either way.

Revisit if the strategy protocol ever grows a sharding-aware seam for another reason."""


def n_devices(mesh: Mesh) -> int:
    return mesh.shape[POP]


def check_population(n: int, mesh: Mesh, *, pairing: int = 1) -> None:
    """Raise unless `n` members shard evenly, loudly and before anything expensive runs.

    `pairing` is the number of members a strategy treats as one indivisible group, and the
    per-device count must be a multiple of it. `Mirrored` declares 2, because it pairs
    members as (2k, 2k+1) and a split landing between them loses the antithetic cancellation
    *silently*: the run still produces an update, just a worse one. With both `n` and the
    device count powers of two this always holds, which is exactly why the one configuration
    where it does not would go unnoticed.

    It used to be `paired: bool` read off `isinstance(strategy, Mirrored)`, which meant a
    user-defined paired strategy could not ask for the alignment it needs. An integer read
    off the strategy covers both and admits groups larger than 2.
    """
    if n < 1:
        raise ValueError(
            f"population must be at least 1, got {n}. `tell` divides by n * sigma, so a "
            "population of zero produces all-NaN parameters rather than an error."
        )
    d = n_devices(mesh)
    if n % d:
        raise ValueError(
            f"population {n} does not divide across {d} devices. shard_map needs an even "
            "split, and an uneven one would change the update rather than fail."
        )
    if pairing > 1 and (n // d) % pairing:
        raise ValueError(
            f"population {n} over {d} devices gives {n // d} members per device, which is "
            f"not a multiple of the strategy's pairing of {pairing}. Mirrored pairs members "
            "as (2k, 2k+1), so a shard that is not a whole number of pairs splits one across "
            "devices and loses the antithetic cancellation without erroring."
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
# bug, and these two are all of them.
#
# There were four. `per_member(mesh, rank)` returned `P(POP, None * rank)` and
# `shard_perturbation(pert, mesh, n)` walked a materialized perturbation placing leaves with
# it. Both are gone, for two independent reasons:
#
#   - `per_member` was a spelling of `members`. JAX pads a short PartitionSpec with None, so
#     `P("pop")` and `P("pop", None, None)` place a rank-3 array identically. Verified for
#     ranks 1, 2 and 3 before removal.
#   - `shard_perturbation` constrained the wrong end. Placing the perturbation does not make
#     the evaluation distribute: the consumer is free to gather it and compute everywhere,
#     and measured, it does. What distributes the evaluation is constraining what `apply`
#     *produces*, which back-propagates and shards the perturbation on its own
#     (`docs/diagnosis-replicated-evaluation.md`).
#
# Neither was ever called from `src/`. They were called only by their own tests, which made
# the suite read as though the perturbation was being placed while nothing placed it.
# --------------------------------------------------------------------------------------


def replicated(mesh: Mesh) -> NamedSharding:
    """Params, and anything else every device needs whole."""
    return NamedSharding(mesh, P())


def members(mesh: Mesh) -> NamedSharding:
    """Anything with a leading member axis: fitness, member ids, a materialized perturbation.

    `P(POP)` with no trailing entries, which is correct at any rank: JAX pads the spec with
    None, so an `(n, episodes)` fitness or an `(n, m, k)` perturbation leaf shards its member
    axis and leaves the rest replicated.
    """
    return NamedSharding(mesh, P(POP))
