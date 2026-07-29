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

    **Cost note.** The chunk loop is Python, so it unrolls at trace time into `n/chunk`
    copies of sample/apply/contract. Jaxpr size and compile time therefore grow linearly
    as `chunk` shrinks: measured, `chunk=1` at `n=128` under a scanning strategy took 87s
    against under a second for `chunk=None`. Pick `chunk` to fit memory, not smaller. If
    a sweep ever needs a small chunk at large n, the fix is a `lax.scan` over chunks,
    which is constant in jaxpr size; the ragged final chunk is the only fiddly part.
    """
    n = int(member_ids.shape[0])

    if chunk is None or chunk >= n:
        pert = strategy.sample(key, params, member_ids)
        fitness = strategy.apply(model, params, pert, sigma)(x)
        update = strategy.contract(pert, shaping(fitness))
    else:
        splits = [member_ids[i : i + chunk] for i in range(0, n, chunk)]

        # Pass one: fitnesses only. n scalars, cheap at any n.
        fitness = jnp.concatenate(
            [
                strategy.apply(model, params, strategy.sample(key, params, s), sigma)(x)
                for s in splits
            ]
        )
        weights = shaping(fitness)

        # Pass two: re-derive each chunk and contract it. Partial contractions summing to
        # the whole is what makes this legal; there is a property test for it.
        update, offset = None, 0
        for s in splits:
            size = int(s.shape[0])
            part = strategy.contract(
                strategy.sample(key, params, s), weights[offset : offset + size]
            )
            update = part if update is None else jax.tree.map(jnp.add, update, part)
            offset += size

    return jax.tree.map(lambda u: u / (n * sigma), update)
