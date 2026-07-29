"""ShardedES: init / ask / eval / tell.

    state = es.init(key, params)              # params is a pytree, no ravel_pytree
    pert, state = es.ask(key, state, n=N)     # shape-aware perturbation for N members
    fitness = evaluate(params, pert)          # user-supplied, shape (N,)
    state = es.tell(state, pert, fitness)

ask returns a Perturbation, not a batch of parameter trees. That is the single most
consequential API decision: returning materialized trees makes LowRank inexpressible,
which is the trap evosax fell into (docs/02-phase1-sharded-core.md C1.1).

Pure functions with explicit state, nothing mutates in place. shard_map requires it.
"""
