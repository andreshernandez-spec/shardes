"""Mesh, NamedSharding, PartitionSpec, and the seed contract.

    from jax import shard_map      # NOT jax.experimental.shard_map, deprecated in JAX 0.8.0

Layout (docs/02-phase1-sharded-core.md C1.2):
  params        replicated across the "pop" axis, every device holds the full model
  perturbations sharded on the member axis, P("pop", None, None)
  fitnesses     sharded, P("pop")
  state         see docs/02 C1.4, the decision is open

Parameters are never sharded. ES's advantage is that every device holds the model and runs
inference independently; sharding parameters reintroduces the communication ES avoids.

The seed contract, which everything else depends on:

    member i's perturbation derives from jax.random.fold_in(base_key, i), where i is the
    global member index. Never the device index, never a per-device counter, never
    sequential consumption of a key stream.

This is what makes device-count invariance and Qiu-style seed regeneration both work, and
it is the single easiest invariant to break by accident.
"""
