"""Strategy A and Strategy B: the two ways to close the ES update loop across devices.

A. Scalar all-reduce, replicated regeneration.
   All-reduce the N fitness scalars (and the N seeds if not derivable, a few KB). Every
   device regenerates all N perturbations from seeds and contracts locally.
   Communication O(N). Contraction replicated D times.

B. Model-size all-reduce of the partial update.
   Each device contracts its local shard into a params-shaped partial, then psum.
   Communication O(d), same as data-parallel SGD. Contraction split D ways.

Both get implemented. The crossover in (N, d, D) is what Phase 2 measures and it is the
strongest single result in the paper (docs/05-paper.md C1).

Nothing public claims "ES only all-reduces scalars" until that measurement exists. The
claim is true for A and false for B, and both are legitimate.

Accumulate over members in f32 even when the perturbations are bf16. Summing 2^18 bf16
terms loses several digits. There is a test for this.

---

**They are not the same shape of computation, and the code reflects that.**

Strategy B is genuinely sharded: each device owns a slice of the population, so its body runs
under `shard_map` in manual mode and ends in a `psum`. Strategy A is not sharded at all. Once
the `N` weights are gathered, every device holds identical inputs and runs identical work, so
it is ordinary replicated computation and writing it as a `shard_map` would be dressing up
something that has no manual axis. `shard_map` in fact refuses it: with `out_specs=P()` it
reports that replication "can't be statically inferred", because nothing in the type system
records that an all-gathered value is the same everywhere.

The observable difference is one collective each, and it is exactly the predicted one:

    A -> all-gather   (N scalars)          and no all-reduce
    B -> all-reduce   (one params-sized)   and no all-gather

`tests/test_contraction.py` asserts that against the compiled HLO rather than against a
comment, which is what makes the O(N)-vs-O(d) claim a measurement.

**Both take `(base_key, member_ids)` and re-derive, rather than a materialized perturbation.**
That is what makes A expressible at all — it *must* regenerate on every device — and it costs
B nothing, because sampling from the seed contract is how every strategy already works. It
also means neither path ever holds an `(N, ...)` array it did not need, which matters at
N = 2^18 where the materialized perturbation is 275 TB.

**They agree to float reassociation, not bitwise.** A sums N terms in one order; B sums N/D
terms per device and then sums D partials. Both are correct and neither is more correct;
`test_strategy_A_equals_B` uses a relative tolerance for that reason, and it is a different
claim from device-count invariance, which *is* near-bitwise because it compares one strategy
against itself.
"""

from __future__ import annotations

import jax
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from shardes.sharding import POP, check_population, replicated
from shardes.strategies.protocol import PerturbationStrategy
from shardes.types import Array, Key, PyTree


def contract_replicated(
    strategy: PerturbationStrategy,
    base_key: Key,
    params: PyTree,
    member_ids: Array,
    weights: Array,
    mesh: Mesh,
) -> PyTree:
    """Strategy A. Gather the N weights, regenerate all N members everywhere, contract.

    Communication is one all-gather of `N` scalars. The contraction is then replicated `D`
    times, which is wasted arithmetic but cheap next to the rollouts that produced the
    fitnesses in the first place.

    Requires seed-derivable perturbations. All three strategies satisfy that by construction,
    which is the seed contract doing work beyond device-count invariance.
    """
    rep = replicated(mesh)
    ids = jax.lax.with_sharding_constraint(member_ids, rep)
    w = jax.lax.with_sharding_constraint(weights, rep)
    return strategy.contract(strategy.sample(base_key, params, ids), w)


def contract_sharded(
    strategy: PerturbationStrategy,
    base_key: Key,
    params: PyTree,
    member_ids: Array,
    weights: Array,
    mesh: Mesh,
) -> PyTree:
    """Strategy B. Each device contracts its own members, then one params-sized psum.

    Communication is one all-reduce the size of the model, the same as data-parallel SGD.
    The contraction is split `D` ways.

    `params` and `base_key` are closed over rather than passed through `in_specs`: both are
    replicated, and threading them as sharded inputs would claim a member axis they do not
    have.
    """

    def local(ids_shard: Array, weights_shard: Array) -> PyTree:
        # Mark the replicated closure as varying across the manual axis before the strategy
        # sees it. A strategy whose `contract` carries a `lax.scan` accumulator derives that
        # accumulator from `params`, and scan requires the carry type to be stable: an
        # accumulator that starts invariant and gains varying-ness from the loop body is
        # rejected. SeedRegenerated is exactly that shape.
        #
        # This is the right home for it. `pcast` needs the axis name, the axis name lives
        # here, and the strategy protocol stays silent about devices — the same reason the
        # mesh is Auto rather than Explicit (see sharding.AXIS_TYPE_NOTE).
        varying = jax.tree.map(lambda x: jax.lax.pcast(x, (POP,), to="varying"), params)
        pert = strategy.sample(base_key, varying, ids_shard)
        partial = strategy.contract(pert, weights_shard)
        return jax.tree.map(lambda leaf: jax.lax.psum(leaf, POP), partial)

    return shard_map(
        local, mesh=mesh, in_specs=(P(POP), P(POP)), out_specs=P()
    )(member_ids, weights)


#: The two strategies by name, so callers and the sweep can select without a conditional.
BY_NAME = {"A": contract_replicated, "B": contract_sharded}


def contract(
    strategy: PerturbationStrategy,
    base_key: Key,
    params: PyTree,
    member_ids: Array,
    weights: Array,
    mesh: Mesh,
    *,
    how: str = "B",
) -> PyTree:
    """Dispatch to A or B. Defaults to B, and the default is provisional.

    B is the safe default because its cost is bounded by the model size, which is known,
    where A's is bounded by the population, which is the thing being scaled. That is a
    reason to default, not a measurement: the crossover in (N, d, D) is Phase 2's job
    (docs/05-paper.md C1), and until it exists nothing here or anywhere claims one wins.
    """
    if how not in BY_NAME:
        raise ValueError(f"contraction strategy must be one of {sorted(BY_NAME)}, got {how!r}")
    check_population(int(member_ids.shape[0]), mesh)
    return BY_NAME[how](strategy, base_key, params, member_ids, weights, mesh)
