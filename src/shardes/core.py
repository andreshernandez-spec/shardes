"""ShardedES: init / ask / eval / tell.

    es = ShardedES(strategy=Mirrored(LowRank(r=1)), n=1024, sigma=0.01, lr=0.05, mesh=mesh)
    state = es.init(key, params)              # params is a pytree, no ravel_pytree
    pert, state = es.ask(state)               # shape-aware perturbation for n members
    fitness = evaluate(es.apply(model, state, pert))    # user-supplied, shape (n,)
    state = es.tell(state, pert, fitness)

`ask` returns a `Perturbation`, not a batch of parameter trees. That is the single most
consequential API decision: returning materialized trees makes LowRank inexpressible, which
is the trap evosax fell into (docs/02-phase1-sharded-core.md C1.1).

Pure functions with explicit state, nothing mutates in place. shard_map requires it.

---

**Two deviations from the signature sketched in docs/02, both deliberate.**

`n` is fixed at construction rather than passed to `ask`. It determines the sharding, and a
population that changes per generation would have to be re-validated against the mesh on
every call — and would silently change what `member_ids` means. Fixing it lets
`check_population` run once, at construction, where the error is attributable.

`ask` takes no key: the base key for generation `g` is `fold_in(state.key, g)`. Passing a
key per call invites a caller to reuse one, which does not raise and does not look wrong —
it just makes two generations sample the same population. Deriving it from a counter in the
state makes that unrepresentable, and keeps `ask` a pure function of state.

**`tell` re-derives rather than trusting the perturbation it is handed.** It reads
`base_key` and `member_ids` off the `Perturbation` — which the protocol requires every
strategy to carry — and hands those to the contraction. Strategy A *must* work this way,
since it regenerates on every device, and Strategy B costs nothing for the consistency. It
also means `tell` never has to hold a materialized `(n, ...)` array.

**The sign convention is descent on the objective, matching `estimator.estimate`.** `tell`
moves `params` *down* the estimated gradient of whatever the fitness measures. Hand it a
loss and it minimizes; hand it a reward and negate first, or it will minimize the reward.
ES implementations that hardcode maximization differ here, so read the sign off `tell` and
not off the word "fitness".
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from shardes import contraction, sharding
from shardes.shaping import centered_ranks
from shardes.strategies._scale import per_leaf
from shardes.strategies.protocol import Perturbation, PerturbationStrategy
from shardes.types import Array, Key, PyTree


class State(NamedTuple):
    """The distribution state, plus what makes the run reproducible.

    `params` is the distribution mean and keeps its pytree structure throughout: nothing
    here is ever flattened (invariant 1).

    `sigma` lives in the state rather than on the object because it is distribution state,
    not configuration — an adaptive rule would change it per generation, and this is where
    that would happen. **Nothing adapts it today**: it is whatever the caller set at `init`,
    for every generation. That is a real limitation rather than an oversight, and
    `docs/BACKLOG.md` B6 records why it was deferred and exactly what it would take.

    It is **either a scalar or a params-shaped pytree**. A scalar is isotropic ES, one global
    step size. A pytree is a per-coordinate diagonal, which is what docs/02 C1.4 settled on
    instead of a sharded CMA state: `sample` produces unit-scale perturbations and `apply`
    scales them, so a diagonal is a type widening on an argument that already existed rather
    than a change to the strategy protocol.

    Stored as given rather than normalized to a tree, so the isotropic case does not pay
    |params| of memory to hold the same number repeated.

    `generation` is the counter the per-generation key derives from, so a resumed run at
    the same generation samples the same population.
    """

    params: PyTree
    sigma: Array
    key: Key
    generation: Array


class ShardedES:
    """ask / eval / tell with the population sharded over the mesh.

    `strategy` is the perturbation scheme. Switching between the two published algorithms
    is a change to this argument and nothing else (docs/02 C1.5):

        ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, ...)          # Qiu et al.
        ShardedES(strategy=Mirrored(LowRank(r=1)), n=262_144, ...)          # EGGROLL

    `how` picks the contraction strategy, A or B (docs/02 C1.3). It changes what is
    communicated and not what is computed; `tests/test_contraction.py` asserts both.
    """

    def __init__(
        self,
        strategy: PerturbationStrategy,
        *,
        n: int,
        sigma: float | PyTree,
        lr: float,
        mesh: Mesh,
        shaping: Callable[[Array], Array] = centered_ranks,
        how: str = "B",
    ):
        # Validated once, here, where the error names the configuration that caused it
        # rather than surfacing as a shape mismatch inside shard_map three calls later.
        # `pairing` is asked of the strategy rather than assumed from its type: a strategy
        # needing an even number of members per device says so, and a pair straddling a
        # device boundary loses the antithetic cancellation silently. This read
        # `isinstance(strategy, Mirrored)`, which worked for the one paired strategy in this
        # repo and gave a user-defined one no way to ask.
        sharding.check_population(n, mesh, pairing=getattr(strategy, "pairing", 1))
        if how not in contraction.BY_NAME:
            raise ValueError(f"how must be one of {sorted(contraction.BY_NAME)}, got {how!r}")

        self.strategy = strategy
        self.n = int(n)
        self.sigma = sigma
        self.lr = float(lr)
        self.mesh = mesh
        self.shaping = shaping
        self.how = how

    # ----------------------------------------------------------------------------------

    def init(self, key: Key, params: PyTree) -> State:
        """Params replicated, everything else scalar. No solution is ever flattened.

        Validates sigma here rather than at construction, because a tree-valued sigma can
        only be checked against `params` and `params` arrives here. `tell` divides by
        `n * sigma`, so a zero or negative leaf produces NaN parameters on the first
        generation with nothing else raising.
        """
        per_leaf(self.sigma, params)  # structure, and its error message
        for leaf in jax.tree.leaves(self.sigma):
            # Concrete leaves only. A traced sigma is legitimate, since sigma lives in the
            # state and an adaptive rule would compute it, and a tracer's value does not
            # exist to compare against zero.
            if isinstance(leaf, (int, float)) or getattr(leaf, "size", 0) and not isinstance(
                leaf, jax.core.Tracer
            ):
                arr = jnp.asarray(leaf)
                if not bool(jnp.all(jnp.isfinite(arr))) or not bool(jnp.all(arr > 0)):
                    raise ValueError(
                        f"sigma must be finite and strictly positive, got {arr}. `tell` "
                        "divides by n * sigma, so a zero or negative step size gives NaN or "
                        "a reversed update rather than an error."
                    )
        return State(
            params=params,
            sigma=jax.tree.map(jnp.float32, self.sigma),
            key=key,
            generation=jnp.int32(0),
        )

    def ask(self, state: State) -> tuple[Perturbation, State]:
        """The perturbation for this generation, and the state advanced past it.

        Returns a `Perturbation`, never materialized parameter trees. Under `LowRank` the
        thing returned is a pair of factors that are never multiplied out; under
        `SeedRegenerated` it is a key and a set of ids and no noise at all.
        """
        base_key = jax.random.fold_in(state.key, state.generation)
        ids = sharding.member_ids(self.n, self.mesh)
        pert = self.strategy.sample(base_key, state.params, ids)
        return pert, state._replace(generation=state.generation + 1)

    def apply(
        self,
        model: Callable[[PyTree, Array], Array],
        state: State,
        pert: Perturbation,
    ) -> Callable[[Array], Array]:
        """`model(params, x) -> y` becomes `g(x) -> (n, ...)`, one output per member.

        The strategy owns the forward pass because the perturbation scheme determines how it
        is structured: full rank materializes per member, low rank rewrites the matmul and
        never does. This owns the *placement*, because the strategies are deliberately
        sharding-agnostic and the mesh lives here.

        **The population is divided by reshaping it into a batch axis, `(D, n/D)`, and
        vmapping over that.** Each row is one device's members. The strategy is handed
        `n/D` ids and evaluates them however it likes; the leading axis is constrained to
        `P("pop")` so GSPMD gives each device one row, and the result is reshaped back to
        `(n, ...)`.

        The reason it is a *batch* axis is the whole design. Everything here turns on which
        shapes GSPMD will partition, and a vmap batch axis is the case it exists to handle.

        **A sharding constraint on the output alone is not enough, and two rented sweeps
        proved it.** Constraining the fitness does distribute a `vmap`, for the reason above.
        It silently does not distribute a `lax.scan`: a scan is a sequential loop with a
        static trip count, so XLA satisfies the constraint by gathering the ids, running all
        `n` iterations on every device, and slicing the result. `SeedRegenerated` evaluates
        with a scan, so it kept the exact `1/D` signature of the original bug straight
        through the fix that repaired the other three strategies, with wall clock flat at
        431.95, 435.29, 436.43, 440.35 ms across `D=1,2,4,8` on 8x A100
        (`docs/diagnosis-seed-regenerated-scan.md`). Reshaping first means the thing being
        partitioned is a batch axis whatever the strategy's body turns out to be.

        **`shard_map` also works and was rejected, which is the more interesting half.** It
        partitions by construction rather than by propagation, so it needs no reshape and
        makes no assumption about the body at all. The cost is that the user's model then
        runs inside a manual mesh, and that makes the manual mesh part of this library's
        public contract: every `lax.scan` and every sharding annotation a user writes in
        their rollout has to be manual-mesh-compatible, with nothing in the API saying so.
        The MuJoCo rollout in `tests/test_control.py` is the first user code to meet that
        requirement and it fails, three tests, on a scan carry that starts invariant inside
        the manual mesh and gains varying-ness from the loop body. Measured: 729 passed and
        3 failed under `shard_map`, 732 passed and 0 failed under this.
        `docs/proposal-scan-strategies-distribute.md` carries all four options and the
        numbers.

        The residual risk is honest and worth stating: this still relies on GSPMD inferring a
        partition, and inference is exactly what failed here. The defence is not that batch
        axes are safe, it is
        `tests/test_sharding.py::test_every_strategy_evaluates_only_its_own_shard`, which
        asserts the property directly for every strategy and fails 15 of 15 without this.

        **Re-deriving rather than slicing the handed perturbation costs nothing, and both
        contractions already work this way.** `contract_replicated` and `contract_sharded`
        both call `strategy.sample(base_key, params, ids)` and use the `Perturbation` only
        for `base_key` and `member_ids`. With this doing the same, a materialized `eps` from
        `ask` has no consumer inside a jitted generation and is eliminated. Regenerability is
        a protocol requirement precisely so this is available.

        `members(mesh)` is `P("pop")` with no trailing entries, which is correct at any rank:
        the `(D, n/D)` ids and an `(D, n/D, episodes)` fitness for `group_relative` both
        shard their leading axis and leave the rest replicated.

        **`x` is replicated to match `params`, and that is a correctness fix rather than a
        convenience.** Every member of a generation must be evaluated on the *same* data,
        common random numbers, or the fitness differences report which member drew an easy
        batch rather than which perturbation was good. Sharding `x` across members would
        break that comparison silently.

        It is also a real footgun without this line: a freshly built batch is committed to
        device 0 while `params` live on the mesh, and any `lax.scan` inside the model that
        carries both raises "Received incompatible devices" from deep inside the user's
        rollout. The MuJoCo adapter hit exactly that. This paragraph described the line for
        some time before the line existed.
        """
        d = sharding.n_devices(self.mesh)

        def row(ids_row: Array, x_row: Array) -> Array:
            """One device's worth of members. `n/D` of them, and the strategy's own shape."""
            local = self.strategy.sample(pert.base_key, state.params, ids_row)
            return self.strategy.apply(model, state.params, local, state.sigma)(x_row)

        def evaluate(x: Array) -> Array:
            x = jax.lax.with_sharding_constraint(x, sharding.replicated(self.mesh))
            # Row-major, so row k is members [k*n/D, (k+1)*n/D), which is the same
            # contiguous split `sharding.member_ids` already hands the mesh. The two have to
            # agree or a member is evaluated under one device's shard and contracted under
            # another's. `check_population` refuses an uneven split at construction, so the
            # reshape cannot be ragged.
            ids = jax.lax.with_sharding_constraint(
                pert.member_ids.reshape(d, self.n // d), sharding.members(self.mesh)
            )
            # in_axes=(0, None): every row sees the whole batch. Common random numbers are
            # the point, so `x` is mapped over nothing.
            out = jax.vmap(row, in_axes=(0, None))(ids, x)
            # Constrained after the vmap as well as before it. The constraint on `ids` says
            # where the work starts; this says where its result lives, and without it the
            # compiler is free to gather the rows back before the reshape.
            out = jax.lax.with_sharding_constraint(out, sharding.members(self.mesh))
            return out.reshape(self.n, *out.shape[2:])

        return evaluate

    def tell(self, state: State, pert: Perturbation, fitness: Array) -> State:
        """Shape the fitnesses, contract into an update, step the mean.

        The shaping is **global and therefore a synchronization barrier**: centered ranks
        need a sort over all `n` fitnesses, which is an all-gather of `n` scalars plus a
        wait, every generation. Cheap in bytes, not free in latency (docs/02 C1.6). Phase 2
        measures what it costs; it is written here as one line so that measurement has
        something unambiguous to point at.

        **The gather stays even though `apply` now pins the fitness sharded**, and it is
        deliberate. It is what lets every shaping be a plain function of an array:
        `centered_ranks` calls `argsort`, `group_relative` calls `mean(axis=0)`, and neither
        can see a mesh. Removing it would either push sharding awareness into the shapings,
        which is the trade `sharding.AXIS_TYPE_NOTE` already declined for the strategies, or
        leave the compiler to insert the gather implicitly and take the barrier back out of
        view. It is no longer load-bearing for the *bug* it once caused: `apply` constrains
        the fitness on the way in, so this cannot propagate backwards past it
        (`docs/diagnosis-replicated-evaluation.md`).

        It is unconditional, so `shaping=none` pays a gather it has no use for. Measured at
        `4N` bytes, 4 KB at `n=1024`, against the 6.29 MB model all-reduce strategy B
        performs at `d=512`. Making it conditional means shapings declaring their
        communication needs, which is a protocol change to save 4 KB.

        For `group_relative` the fitness is `(n, g)` and the gather is `4Ng` rather than
        `4N`, so the barrier grows with the task count. It only matters where `Ng` approaches
        the parameter count: at `d=512` that is `Ng = 1.57e6`, or 1536 tasks at `n=1024`.
        A `psum` of the per-task statistics would be `O(g)` instead, and would cost the same
        sharding awareness this paragraph just declined.

        The `1/(n*sigma)` factor lives here rather than in `contract`, so partial
        contractions over disjoint members still sum to the whole. That is what makes both
        chunking and Strategy B valid.
        """
        weights = self.shaping(
            jax.lax.with_sharding_constraint(fitness, sharding.replicated(self.mesh))
        )
        # The shaping contract is "leading axis is members, returns (n,)", and nothing
        # enforced it until this check. A 1-D shaping handed an (n, episodes) fitness does
        # not raise: `centered_ranks` ranks along the last axis and hands back (n, episodes),
        # which then fails inside `contract`'s einsum with a message about subscript 'n' that
        # says nothing about episodes or shaping. Caught here, where the fix is obvious:
        # reduce the episode axis yourself, or use a shaping that consumes it
        # (`group_relative`).
        if weights.shape != (self.n,):
            raise ValueError(
                f"shaping returned {weights.shape}, expected {(self.n,)}. A fitness with "
                "more than one axis needs a shaping that reduces the extra axes "
                "(shardes.shaping.group_relative), or reduce them before calling tell."
            )
        weights = jax.lax.with_sharding_constraint(weights, sharding.members(self.mesh))

        update = contraction.contract(
            self.strategy,
            pert.base_key,
            state.params,
            pert.member_ids,
            weights,
            self.mesh,
            how=self.how,
        )
        # Per leaf, because sigma may be a params-shaped diagonal. The estimator divides by
        # the sigma it perturbed with, so a per-coordinate sigma divides per coordinate.
        sigmas = per_leaf(state.sigma, state.params)
        return state._replace(
            params=jax.tree.map(
                lambda p, s, u: p - (self.lr / (self.n * s)) * u,
                state.params, sigmas, update,
            )
        )
