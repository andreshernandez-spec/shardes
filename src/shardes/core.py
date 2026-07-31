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
from shardes.strategies.mirrored import Mirrored
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
        # `paired` is asked of the strategy rather than assumed: Mirrored needs an even
        # number of members per device or a pair straddles a device boundary and the
        # antithetic cancellation is silently lost.
        sharding.check_population(n, mesh, paired=isinstance(strategy, Mirrored))
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
        """Params replicated, everything else scalar. No solution is ever flattened."""
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

        The strategy owns this because the perturbation scheme determines how the forward
        pass is structured: full rank materializes per member, low rank rewrites the matmul
        and never does.

        **`x` is replicated to match `params`, and that is a correctness fix rather than a
        convenience.** Every member of a generation must be evaluated on the *same* data —
        common random numbers — or the fitness differences report which member drew an easy
        batch rather than which perturbation was good. Sharding `x` across members would
        break that comparison silently.

        It is also a real footgun without this line. `init` commits `params` to the mesh,
        while a freshly built batch is committed to device 0, and any `lax.scan` inside the
        model that carries both raises "Received incompatible devices" from deep inside the
        user's rollout. The MuJoCo adapter hit exactly that.
        """
        return self.strategy.apply(model, state.params, pert, state.sigma)

    def tell(self, state: State, pert: Perturbation, fitness: Array) -> State:
        """Shape the fitnesses, contract into an update, step the mean.

        The shaping is **global and therefore a synchronization barrier**: centered ranks
        need a sort over all `n` fitnesses, which is an all-gather of `n` scalars plus a
        wait, every generation. Cheap in bytes, not free in latency (docs/02 C1.6). Phase 2
        measures what it costs; it is written here as one line so that measurement has
        something unambiguous to point at.

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
