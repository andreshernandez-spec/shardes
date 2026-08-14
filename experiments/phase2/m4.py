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

Their library ships RWKV models, but their `Noiser` is not tied to them: `do_mm`, `do_Tmm`
and `do_emb` are the same three seams as this project's `dense` and `embed`, down to the
transpose. So the arm is their `EggRoll` unmodified, driven by a forward pass written here
that routes this project's transformer block through those seams. No adapter, no port, and
nothing that has to track their model code.

Installing them is the awkward part, not calling them: `hyperscalees/__init__.py` pulls a
model zoo needing torch, gymnax, distrax, transformers and more, while the `noiser` package
being benchmarked needs only `jax` and `optax`. See `_import_eggroll`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
import types
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

#: Cache for `_eggroll_or_reason`. Their loader has side effects in `sys.modules`.
_EGGROLL = None
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


def _import_eggroll():
    """Their `EggRoll`, unmodified, from the installed package. Never a copy.

    The ordinary import is tried first and used when it works. It usually does not, and the
    reason is packaging rather than anything about the algorithm: `hyperscalees/__init__.py`
    imports their model zoo, which needs gymnax, distrax, transformers, datasets, torch and
    six more. The `noiser` subpackage that this benchmark actually exercises imports `jax`,
    `optax` and stdlib, and nothing else. Checked at `b77f7d6`, every import in
    `src/hyperscalees/noiser/*.py`.

    So the fallback loads their two noiser modules from the installed package directory with
    a package skeleton registered first, which makes `from .base_noiser import Noiser`
    resolve. Their code runs unmodified; only the route to it skips `__init__`.

    **This is a decision with two defensible answers and it is flagged in `docs/03`.** The
    other is to install their full dependency set. That was rejected because it puts torch's
    CUDA stack beside JAX's on the benchmark node, and a throughput measurement is the wrong
    place to find out whether those two coexist. The cost is that this reaches past a
    package boundary, so it breaks if they move the file. It breaks loudly.
    """
    try:
        from hyperscalees.noiser.eggroll import EggRoll
        return EggRoll, "package import"
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is not None and missing.startswith("hyperscalees"):
            raise  # their package is broken or absent, not merely under-installed

    spec = importlib.util.find_spec("hyperscalees")
    if spec is None or not spec.origin:
        raise ModuleNotFoundError("hyperscalees")
    root = Path(spec.origin).parent
    for name, path in (("hyperscalees", root), ("hyperscalees.noiser", root / "noiser")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            # A stub without `__spec__` makes a later `find_spec("hyperscalees")` raise
            # `ValueError: __spec__ is None` rather than answering the question. The second
            # call into this arm did exactly that.
            mod.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            mod.__spec__.submodule_search_locations = [str(path)]
            sys.modules[name] = mod
    loaded = None
    for leaf in ("base_noiser", "eggroll"):  # base first: eggroll imports it
        full = f"hyperscalees.noiser.{leaf}"
        s = importlib.util.spec_from_file_location(full, root / "noiser" / f"{leaf}.py")
        loaded = importlib.util.module_from_spec(s)
        sys.modules[full] = loaded
        s.loader.exec_module(loaded)
    return loaded.EggRoll, f"direct module load ({missing} missing, model zoo not installed)"


def _eggroll_or_reason():
    """`_import_eggroll` once, cached, with the failure turned into a printable reason.

    Cached because `main` probes availability and then builds the arm per shape, and the
    load registers modules in `sys.modules` as a side effect.
    """
    global _EGGROLL
    if _EGGROLL is None:
        try:
            _EGGROLL = _import_eggroll()
        except ModuleNotFoundError as exc:
            _EGGROLL = ("unavailable", f"{exc.name} not importable "
                                       "(hyperscalees is GPL-3.0, never vendored here)")
        except Exception as exc:  # noqa: BLE001 - the failure itself is the datum
            _EGGROLL = ("unavailable", f"{type(exc).__name__}: {exc}")
    return _EGGROLL


def _eggroll_forward(eggroll, frozen, noiser, params, keys, iterinfo, x):
    """`transformer_block.forward`, with every matmul routed through their `do_mm`.

    Their `Noiser` is model-agnostic: `do_mm` is `x @ W.T` plus the member's rank-`r`
    correction, computed from the key and never materialized. That is the same contract as
    this project's `shardes.nn.dense`, down to the transpose, so this is a rewrite of six
    call sites rather than an adapter.

    `_rms_norm` is reused from `transformer_block` rather than copied. A copy could drift,
    and then the two arms would be timing different models while appearing to time one.
    """
    def mm(name, h):
        return eggroll.do_mm(frozen, noiser, params[name], keys[name], iterinfo, h)

    d = x.shape[-1]
    h = transformer_block._rms_norm(x)
    q, k, v = mm("wq", h), mm("wk", h), mm("wv", h)
    scores = jnp.einsum("bqd,bkd->bqk", q, k) * d**-0.5
    attn = jax.nn.softmax(scores, axis=-1)
    x = x + mm("wo", jnp.einsum("bqk,bkd->bqd", attn, v))

    h = transformer_block._rms_norm(x)
    return x + mm("w_down", jax.nn.gelu(mm("w_up", h)))


def arm_eggroll(shape: Shape, mesh):
    """EGGROLL's own implementation, driven at this project's shapes.

    Not one line of `hyperscalees` is copied here; see the module docstring on GPL-3.0.
    `_import_eggroll` loads their `EggRoll` and this drives it.

    **Three choices that decide whether the comparison is fair, stated here because none of
    them is visible in the number:**

    `rank=1` matches `LowRank(r=1)`, and `es_map` marks all six matrices `MM_PARAM` so every
    one takes their low-rank path. Nothing is frozen and nothing falls back to a dense
    perturbation, which would quietly change what is being timed.

    `noise_reuse=1` gives fresh noise each generation. **Their default of 0 means reuse
    forever**, not "no reuse": `true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse`,
    so a default-constructed noiser evaluates the same perturbations every generation. Their
    own experiment scripts pass the flag. Timing barely moves either way; the point is that
    a reader assumes fresh sampling and would be wrong.

    **Their scheme is antithetic by construction**, `thread_id // 2` with the sign from
    `thread_id % 2`, so `N` members are `N/2` directions. The matched arm on this side is
    `mirrored_lr1`, not `lowrank_r1`. Pairing it against the unmirrored arm would compare
    `N` directions with `N/2` and read as a throughput result.

    Fitness is `-loss`: they maximize, this project minimizes. It does not affect the timing
    and it keeps the arm descending rather than diverging.
    """
    got = _eggroll_or_reason()
    if got[0] == "unavailable":
        return got
    eggroll, how = got

    n = shape.population
    if n % 2:
        return ("unavailable", f"population {n} is odd; their antithetic pairing needs even")

    import optax  # noqa: PLC0415 - optional, like the arm itself

    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=shape.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=shape.d_model, batch=shape.batch, seq=shape.seq
    )
    names = sorted(params)
    keys = dict(zip(names, jax.random.split(jax.random.fold_in(key, 2), len(names))))
    es_map = {name: 1 for name in params}  # MM_PARAM: their low-rank path, per models/common

    frozen, noiser_params = eggroll.init_noiser(
        params, SIGMA, LR, solver=optax.sgd, rank=1, noise_reuse=1
    )

    @jax.jit
    def step(state):
        noiser, params_, gen = state
        iterinfos = (jnp.full(n, gen, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32))

        def member(epoch, thread_id):
            out = _eggroll_forward(
                eggroll, frozen, noiser, params_, keys, (epoch, thread_id), batch.x
            )
            return jnp.mean(jnp.square(out - batch.target))

        losses = jax.vmap(member)(*iterinfos)
        fitness = eggroll.convert_fitnesses(frozen, noiser, -losses)
        noiser, params_ = eggroll.do_updates(
            frozen, noiser, params_, keys, fitness, iterinfos, es_map
        )
        return noiser, params_, gen + 1

    step.eggroll_how = how
    return step, (noiser_params, params, jnp.int32(0))


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
    # `eggroll` sits next to `mirrored_lr1` deliberately: their scheme is antithetic by
    # construction, so that is the arm with the same rank and the same direction count.
    arms = [("shardes/mirrored_lr1/B", lambda s: arm_shardes(s, "mirrored_lr1", "B", mesh), True),
            ("shardes/seed_regenerated/B", lambda s: arm_shardes(s, "seed_regenerated", "B", mesh), True),
            ("eggroll/rank1", lambda s: arm_eggroll(s, mesh), False),
            ("naive_es", lambda s: arm_naive(s, mesh), False),
            ("evosax/Open_ES", lambda s: arm_evosax(s, mesh), False)]

    if args.dry_run:
        for s in shapes:
            print(f"  d={s.d_model} N={s.population} params={s.n_params:,} "
                  f"tokens/gen={s.tokens:,}")
        print(f"\n{len(shapes)} shape(s) x {len(arms)} arm(s)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    # `--out` is an output whatever it is called, so it does not count against the worktree
    # being clean. `run.py` had the same bug: a directory it wrote but had not declared was
    # counted as a foreign untracked file, and every record stamped itself unreproducible.
    env = harness.capture_env(HERE, (*OUTPUTS, args.out.name))
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
