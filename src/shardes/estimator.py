"""The ES gradient estimator: sample -> apply -> shape -> contract -> scale.

    g_hat = (1 / (n * sigma)) * sum_i shaping(fitness)_i * eps_i

**Sign convention: no flip.** `model` returns the objective being differentiated and
`estimate` returns an estimate of `grad f` of that objective. Hand it a loss and you get
`grad loss`; hand it a reward and you get an ascent direction. ES implementations that
hardcode maximization differ here, so read the sign off this line rather than assuming.

Unbiasedness: for a quadratic the estimator is exact at *any* sigma, because
`E[f(theta + sigma eps) eps] = sigma H theta` with the third moment vanishing. That is why
the quadratic is the unbiasedness oracle (docs/conventions.md).
"""

from typing import Callable

import jax
import jax.numpy as jnp

from shardes.strategies.protocol import PerturbationStrategy
from shardes.types import Array, Key, PyTree


def estimate(
    strategy: PerturbationStrategy,
    model: Callable[[PyTree, Array], Array],
    params: PyTree,
    x: Array,
    key: Key,
    *,
    member_ids: Array,
    sigma: float,
    shaping: Callable[[Array], Array],
    chunk: int | None = None,
) -> PyTree:
    """Estimate grad f at `params`, params-shaped.

    member_ids: (n,) global member indices.
    chunk:      None keeps the whole perturbation and contracts once. An int streams in
                chunks of that size.

    `chunk` is not only a memory knob, it picks the contraction strategy
    (docs/02-phase1-sharded-core.md C1.3):

    - `chunk=None` keeps the perturbation and contracts once. **Strategy B** in miniature.
    - an int re-derives each chunk from `(key, member_ids)` in a second pass.
      **Strategy A** in miniature.

    Two passes are unavoidable when chunking, because shaping is global: centered ranks
    need all n fitnesses before any weight is known. Pass one collects n scalars, which is
    cheap to hold at any n; pass two re-samples and contracts. That both paths must agree
    is `test_chunked_matches_unchunked`, which is `test_strategy_A_equals_strategy_B` from
    Phase 1, available now on one device.

    The chunk loop is a `lax.scan`, so jaxpr size and compile time are constant in
    `n/chunk`. An earlier Python loop unrolled one traced sample/apply/contract per chunk
    and `chunk=1` at `n=128` under a scanning strategy took 87s to compile against under a
    second for `chunk=None`. Pick `chunk` to fit memory now, not to keep the trace small.

    A ragged final chunk is handled by padding `member_ids` up to a whole number of chunks
    and zeroing the padded weights. The padded members' *fitnesses* are computed and
    discarded, which is the one wasted slice; their contributions to the update are
    exactly zero because a weight of zero is exact in floating point.
    """
    n = int(member_ids.shape[0])

    if chunk is None or chunk >= n:
        pert = strategy.sample(key, params, member_ids)
        fitness = strategy.apply(model, params, pert, sigma)(x)
        update = strategy.contract(pert, shaping(fitness))
    else:
        n_chunks = -(-n // chunk)  # ceil
        pad = n_chunks * chunk - n

        # Pad with a repeat of member 0. Any valid id works: padded slots carry weight
        # zero in pass two, and reusing a real id keeps `sample` on shapes it already sees.
        ids = member_ids
        if pad:
            ids = jnp.concatenate([ids, jnp.full((pad,), ids[0], ids.dtype)])
        ids = ids.reshape(n_chunks, chunk)

        # Pass one: fitnesses only. n scalars, cheap at any n.
        def fitness_step(carry, chunk_ids):
            pert = strategy.sample(key, params, chunk_ids)
            return carry, strategy.apply(model, params, pert, sigma)(x)

        _, stacked = jax.lax.scan(fitness_step, None, ids)
        fitness = stacked.reshape((n_chunks * chunk,) + stacked.shape[2:])[:n]

        weights = shaping(fitness)
        if pad:
            weights = jnp.concatenate([weights, jnp.zeros((pad,), weights.dtype)])

        # Pass two: re-derive each chunk and contract it. Partial contractions summing to
        # the whole is what makes this legal; there is a property test for it.
        #
        # The carry's shape and dtype come from tracing one chunk rather than being
        # assumed params-shaped f32, so a strategy contracting to something else still
        # scans. lax.scan is strict about carry structure and would fail late otherwise.
        spec = jax.eval_shape(
            lambda i, w: strategy.contract(strategy.sample(key, params, i), w),
            ids[0],
            weights[:chunk],
        )
        init = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), spec)

        def contract_step(acc, ids_and_weights):
            chunk_ids, chunk_weights = ids_and_weights
            part = strategy.contract(strategy.sample(key, params, chunk_ids), chunk_weights)
            return jax.tree.map(jnp.add, acc, part), None

        update, _ = jax.lax.scan(
            contract_step, init, (ids, weights.reshape(n_chunks, chunk))
        )

    return jax.tree.map(lambda u: u / (n * sigma), update)
