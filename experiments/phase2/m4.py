"""M4: throughput against external references, at matched shapes.

    python m4.py --dry-run                       # what would run, and which arms are present
    python m4.py --d-model 512 --population 256  # one shape, every available arm
    python m4.py --config sweep-postfix.yaml     # the sweep's shapes

`docs/03` asks for naive ES, EGGROLL's own implementation, and optionally evosax, reported as
tokens/s at a stated parameter count. This runs the arms that are importable and says which
were not, rather than silently comparing against fewer references than the reader expects.

**The claim being tested is throughput at matched shapes, not solution quality.** Every arm
gets the same model, the same population, the same batch and the same number of generations,
and the number reported is tokens per second. Whether an arm's update rule finds a better
optimum is a different experiment, and `docs/01` is where the estimator quality lives. Two
consequences worth stating before anyone quotes a ratio:

  - evosax's `Open_ES` applies an optax optimizer (Adam by default) and shardes applies plain
    SGD. This uses SGD on both sides where the API allows it, and the residual difference is
    a few elementwise ops on a parameter-sized array, which is not where the time goes.
  - Nothing here checks that two arms take the same *step*. They do not: antithetic pairing,
    rank shaping and the estimator's normalisation all differ in detail. Reporting tokens/s
    is the honest scope.

**Tokens, not generations.** `tokens = population * batch * seq`, the count of model forward
passes' worth of data per generation, which is how both papers report and the only unit that
survives comparing a `population=256` arm with a `population=1024` arm.

---

**On EGGROLL: `hyperscalees` is GPL-3.0 and `shardes` is Apache-2.0.**

`ESHyperscale/HyperscaleES` is the authors' implementation. Not one line of it is copied
here, and none should be. Vendoring GPL-3.0 source into an Apache-2.0 repository relicenses
the result, and a benchmark is not worth that. The arm below imports the package if the
person running the benchmark installed it, exactly as the evosax arm does, and skips
otherwise. That is the same shape as `CLAUDE.md`'s "evosax is a comparison target, not a
dependency", and it is a licence boundary rather than a packaging preference: do not
"simplify" it later by copying their code in.

Their library is built for RWKV language models, so an arm that drives it at this project's
transformer-block shapes is adaptation work rather than an import. It is stubbed until that
is done, and `--dry-run` says so.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.iid_gaussian import IIDGaussian  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

OUTPUTS = ("results-m4", "env.json")
SEED, SIGMA, LR = 0, 0.01, 0.05


@dataclass
class Shape:
    d_model: int
    population: int
    batch: int = 8
    seq: int = 32

    @property
    def tokens(self) -> int:
        """Model-forward tokens per generation. The unit both papers report."""
        return self.population * self.batch * self.seq

    @property
    def n_params(self) -> int:
        """Six (d, d) matrices. Stated with every number, per docs/03."""
        return 6 * self.d_model * self.d_model


# ------------------------------------------------------------------------------------
# Arms. Each returns a jitted `step(state) -> state` plus a name, or None if unavailable.
# ------------------------------------------------------------------------------------


def arm_shardes(shape: Shape, strategy_name: str, how: str, mesh):
    """This library. The strategy is named so the table can show more than one."""
    make = {
        "iid_gaussian": IIDGaussian,
        "seed_regenerated": SeedRegenerated,
        "lowrank_r1": lambda: LowRank(r=1),
        "mirrored_lr1": lambda: Mirrored(LowRank(r=1)),
        "mirrored_seed": lambda: Mirrored(SeedRegenerated()),
    }[strategy_name]
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=shape.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=shape.d_model, batch=shape.batch, seq=shape.seq
    )
    es = ShardedES(make(), n=shape.population, sigma=SIGMA, lr=LR, mesh=mesh, how=how)

    @jax.jit
    def step(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness)

    return step, es.init(key, params)


def arm_naive(shape: Shape, mesh):
    """Naive ES: materialize every perturbation as a full parameter tree, then vmap.

    **The baseline both papers beat, and it is deliberately not clever.** Every member gets
    its own copy of every parameter, so auxiliary memory is `n * |params|` and the forward
    pass cannot share a base GEMM. That is the thing EGGROLL's low-rank factorisation and
    Qiu's seed regeneration each remove, so a comparison that quietly optimised it would be
    measuring a strawman nobody proposed.

    It lives here rather than in `src/` because it is a reference point, not a strategy
    anyone should use. `IIDGaussian` is the library's own materializing path and is a
    different thing: it materializes the *perturbation*, shards it, and never builds `n`
    parameter trees.
    """
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=shape.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=shape.d_model, batch=shape.batch, seq=shape.seq
    )
    n = shape.population
    leaves, treedef = jax.tree.flatten(params)

    @jax.jit
    def step(state):
        params_, gen = state
        gkey = jax.random.fold_in(jax.random.key(SEED), gen)
        keys = jax.random.split(gkey, n)

        def one_eps(k):
            ks = jax.random.split(k, len(leaves))
            return jax.tree.unflatten(
                treedef, [jax.random.normal(kk, l.shape, l.dtype) for kk, l in zip(ks, leaves)]
            )

        eps = jax.vmap(one_eps)(keys)  # (n, ...) per leaf: the whole point of "naive"
        perturbed = jax.tree.map(lambda p, e: p + SIGMA * e, params_, eps)
        fitness = jax.vmap(transformer_block.loss, in_axes=(0, None))(perturbed, batch)
        # Centred ranks, matching shardes' default shaping so the arithmetic per generation
        # is comparable rather than merely the shapes.
        ranks = jnp.argsort(jnp.argsort(fitness)).astype(jnp.float32)
        w = ranks / (n - 1) - 0.5
        update = jax.tree.map(lambda e: jnp.tensordot(w, e, axes=1), eps)
        params_ = jax.tree.map(lambda p, u: p - (LR / (n * SIGMA)) * u, params_, update)
        return params_, gen + 1

    return step, (params, jnp.int32(0))


def arm_evosax(shape: Shape, mesh):
    """evosax `Open_ES`, the incumbent. Flattens every solution via `ravel_pytree`.

    Returns None if evosax is not installed, which is a legitimate state: `CLAUDE.md` keeps
    it a comparison target rather than a dependency.

    The flattening is the architectural difference this project exists to avoid, so this arm
    is the one whose *memory* is as interesting as its throughput. It is also why it may fail
    at populations the other arms handle: that failure is a result, and it is recorded rather
    than caught and hidden.
    """
    try:
        from evosax.algorithms import Open_ES
    except ImportError:
        return None

    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=shape.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=shape.d_model, batch=shape.batch, seq=shape.seq
    )
    try:
        import optax
        solver = Open_ES(
            population_size=shape.population,
            solution=params,
            optimizer=optax.sgd(LR),
        )
    except Exception as exc:  # noqa: BLE001 - the failure itself is the datum
        return ("unavailable", f"{type(exc).__name__}: {exc}")

    es_params = solver.default_params
    state = solver.init(key, params, es_params)

    @jax.jit
    def step(state):
        k1, k2 = jax.random.split(jax.random.fold_in(key, state.generation_counter), 2)
        population, state = solver.ask(k1, state, es_params)
        fitness = jax.vmap(transformer_block.loss, in_axes=(0, None))(population, batch)
        state, _ = solver.tell(k2, population, fitness, state, es_params)
        return state

    return step, state


def arm_eggroll(shape: Shape, mesh):
    """EGGROLL's own implementation. Not wired up; see the module docstring on GPL-3.0.

    Returns a reason rather than None so `--dry-run` can print why the most valuable arm in
    `docs/03` is missing instead of leaving a blank the reader has to interpret.
    """
    if importlib.util.find_spec("hyperscalees") is None:
        return ("unavailable", "hyperscalees not installed (GPL-3.0, never vendored here)")
    return ("unavailable", "installed, but no adapter yet: their API targets RWKV, not this "
                           "transformer block. Adaptation work, not an import.")


# ------------------------------------------------------------------------------------


def _reason(built) -> str | None:
    """Why an arm is missing, or None if it is present.

    An arm builds to `(step, state)` and an absent one to `("unavailable", reason)`, so the
    two cannot be told apart by `isinstance(v, tuple)`. They were, once, which serialized an
    optimizer state into the results file.
    """
    if built is None:
        return "not installed"
    if isinstance(built, tuple) and len(built) == 2 and built[0] == "unavailable":
        return built[1]
    return None


def timed(step: Callable, state, warmup: int, repeats: int) -> tuple[float, float]:
    """Median and IQR seconds per generation. Same guards as `run.py`.

    Warm-up discarded because the first call compiles, and `block_until_ready` on every
    result because otherwise this times async dispatch and nothing else.
    """
    for _ in range(warmup):
        state = step(state)
    jax.block_until_ready(state)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        state = step(state)
        jax.block_until_ready(state)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    q = statistics.median
    return q(samples), q(samples[len(samples) // 2:]) - q(samples[: (len(samples) + 1) // 2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=None,
                    help="take shapes from a sweep config instead of --d-model/--population")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "results-m4")
    args = ap.parse_args(argv)

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        shapes = [
            Shape(d, n, cfg.get("batch", 8), cfg.get("seq", 32))
            for d in cfg["d_model"]
            for n in cfg["population"][d]
        ]
    else:
        shapes = [Shape(args.d_model, args.population, args.batch, args.seq)]

    mesh = sharding.make_mesh(args.devices)
    d = sharding.n_devices(mesh)
    print(f"{len(jax.devices())} x {jax.devices()[0].device_kind}, mesh of {d}\n")

    # Which arms exist, stated once and up front. A missing reference is a caveat on the
    # whole measurement, not a footnote on one row.
    probe = Shape(64, 8, 2, 4)
    availability = {
        "evosax": arm_evosax(probe, sharding.make_mesh(1)),
        "eggroll": arm_eggroll(probe, sharding.make_mesh(1)),
    }
    for name, got in availability.items():
        why = _reason(got)
        print(f"  {name}: available" if why is None else f"  {name}: unavailable, {why}")
    print()

    # `sharded` says whether the arm uses more than one device. **Only shardes does.**
    # naive ES and evosax have no sharding path at all, so at `--devices 8` they still run on
    # one, and a table that put their number beside a shardes-on-8 number without saying so
    # would be comparing a library against a library plus seven idle GPUs. `docs/03` asks for
    # "same GPU, same shapes, same N", so the primary comparison is `--devices 1`; the D=8
    # column shows what sharding adds and is not a like-for-like ratio.
    arms = [("shardes/mirrored_lr1/B", lambda s: arm_shardes(s, "mirrored_lr1", "B", mesh), True),
            ("shardes/seed_regenerated/B", lambda s: arm_shardes(s, "seed_regenerated", "B", mesh), True),
            ("naive_es", lambda s: arm_naive(s, mesh), False),
            ("evosax/Open_ES", lambda s: arm_evosax(s, mesh), False)]

    if args.dry_run:
        for s in shapes:
            print(f"  d={s.d_model} N={s.population} params={s.n_params:,} "
                  f"tokens/gen={s.tokens:,}")
        print(f"\n{len(shapes)} shape(s) x {len(arms)} arm(s)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    env = harness.capture_env(HERE, OUTPUTS)
    rows = []
    if d > 1:
        print("NOTE: only the shardes arms use the mesh. naive_es and evosax run on one\n"
              "device whatever --devices says, so their rows are not a like-for-like\n"
              "comparison at D>1. Use --devices 1 for that.\n")
    print(f"{'shape':>22}  {'arm':28}{'ms/gen':>10}{'+-':>8}{'tokens/s':>14}{'mesh':>7}")
    for s in shapes:
        for name, build, sharded in arms:
            built = build(s)
            if _reason(built) is not None:
                continue
            try:
                step, state = built
                med, iqr = timed(step, state, args.warmup, args.repeats)
                toks = s.tokens / med
                print(f"d={s.d_model:>5} N={s.population:>5}  {name:28}"
                      f"{med * 1e3:>10.2f}{iqr * 1e3:>8.2f}{toks:>14,.0f}"
                      f"{('yes' if sharded else 'no'):>7}")
                rows.append({"d_model": s.d_model, "population": s.population,
                             "batch": s.batch, "seq": s.seq, "n_params": s.n_params,
                             "arm": name, "seconds_median": med, "seconds_iqr": iqr,
                             "tokens_per_second": toks, "devices": d,
                             "sharded": sharded, "env": env})
            except Exception as exc:  # noqa: BLE001
                # An arm that cannot run this shape is a result. evosax flattening itself
                # out of memory at a population the others handle is exactly the finding.
                print(f"d={s.d_model:>5} N={s.population:>5}  {name:28}{'FAILED':>10}  "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
                rows.append({"d_model": s.d_model, "population": s.population,
                             "arm": name, "failed": f"{type(exc).__name__}: {exc}",
                             "devices": d, "sharded": sharded, "env": env})

    # `_reason` rather than a bare tuple check: a built arm is also a tuple, `(step, state)`,
    # and evosax's state is not JSON serializable. Conflating the two wrote the optimizer's
    # State into the results file and crashed the run after every timing had been taken.
    out = args.out / f"m4-d{d}.json"
    harness.write_atomic(out, {"rows": rows, "env": env,
                               "unavailable": {k: _reason(v) for k, v in availability.items()
                                               if _reason(v) is not None}})
    print(f"\n{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
