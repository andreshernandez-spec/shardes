#!/usr/bin/env python
"""Where the wall clock goes inside one generation, at D=1 against D=8.

    python profile.py --config sweep.yaml              # the table
    python profile.py --config sweep.yaml --devices 8  # one device count

The measurement `docs/03` needs to close G2 criterion 1. M1 measured 0.112 to 0.142 parallel
efficiency and M5 ruled out communication volume (0.3% of NVLink at the worst configuration),
so the gate's "worse than expected is fine if you identify the cause" clause is still open.
This is the attribution.

**Four timings per configuration, each a complete jitted program.**

- `full`      the generation the sweep timed: ask, apply, tell, with `centered_ranks`.
- `no_shape`  the same with `shaping=none`, which is the only shaping that is not a global
              barrier (`src/shardes/shaping.py`). `full - no_shape` is what the shaping
              barrier costs, sort and synchronization together.
- `eval`      ask and apply only, returning the fitness. This is the part that *should*
              fall as `1/D` under strong scaling, because it is the population evaluation
              and the population is sharded.
- `floor`     a jitted `psum` of one scalar over the same mesh, repeated. Nothing to compute,
              so this is dispatch plus one synchronization: the smallest a generation on
              this mesh could possibly be.

`full - eval` is what `tell` costs. Each figure is a separate compilation, so the parts do
not sum to the whole and are not meant to: XLA fuses differently when the program changes.
What is comparable is **how each one moves with D**, which is the question.

**The FLOPs columns answer it without a stopwatch, and they are the result.** XLA reports
the cost of the compiled *per-device* program, so under SPMD a computation that distributes
has its per-device FLOPs fall as `1/D`. Measured on the sweep's shapes, `eval` FLOPs are
**identical** at `D = 1, 2, 4, 8`, for every strategy and both contractions: every device
evaluates the whole population. That is the flat scaling curve, and it explains the
magnitude and not just the direction. If per-device work does not fall, `T_D` equals `T_1`
and parallel efficiency `T_1/(D T_D)` is exactly `1/D`. M1 measured 0.112 to 0.142 at
`D = 8` against `1/8 = 0.125`.

FLOPs are a property of the compiled program, so **the static columns are valid on
simulated devices** and this part needs no rented GPU. Validated against `N`: eval FLOPs at
`d=512` are 246.4e9, 492.9e9 and 985.7e9 for `N` of 256, 512 and 1024, exactly linear.

**Timing needs real devices.** `--xla_force_host_platform_device_count` gives eight devices
that share memory and never communicate, so the millisecond columns would be fiction
(`docs/06`). Run those where the sweep ran; the FLOPs columns stand either way.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import shard_map  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

import run as R  # noqa: E402  SEED, SIGMA, LR, STRATEGIES, so they cannot drift

from shardes import sharding, shaping  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402


def timed(fn, state, warmup: int, repeats: int) -> tuple[float, float]:
    """Median and IQR seconds per call, with `run.py`'s guards: warm-up discarded, blocked on.

    **The IQR is not decoration here.** `tell` and `shaping` are differences between
    separately compiled programs, so when the part being isolated is smaller than the
    run-to-run variation the difference comes out negative. That is noise announcing itself,
    and it has to be visible in the table or it reads as a measurement.
    """
    for _ in range(warmup):
        state_out = fn(state)
    jax.block_until_ready(state_out)
    took = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(state))
        took.append(time.perf_counter() - t0)
    took.sort()
    q1, q3 = took[len(took) // 4], took[(3 * len(took)) // 4]
    return statistics.median(took), q3 - q1


def floor_seconds(mesh, warmup: int, repeats: int) -> float:
    """One `psum` of a scalar over the mesh. Dispatch plus a synchronization, nothing else."""
    @jax.jit
    def once(x):
        return shard_map(
            lambda y: jax.lax.psum(y, "pop"), mesh=mesh, in_specs=P("pop"), out_specs=P()
        )(x)

    x = jnp.ones((mesh.shape["pop"],), jnp.float32)
    return timed(once, x, warmup, repeats)[0]


def flops_of(fn, state) -> float:
    """FLOPs in the compiled per-device program. Under SPMD a computation that distributes
    has this fall as 1/D; one that does not, does not."""
    with jax.default_matmul_precision("highest"):
        analysis = jax.jit(fn).lower(state).compile().cost_analysis()
    analysis = analysis[0] if isinstance(analysis, list) else analysis
    return float(analysis.get("flops", float("nan")))


def profile(d_model: int, population: int, strategy: str, how: str, devices: int,
            batch: int, seq: int, warmup: int, repeats: int, static: bool = False) -> dict:
    mesh = sharding.make_mesh(devices)
    key = jax.random.key(R.SEED)
    params = transformer_block.init(key, d_model=d_model)
    data = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=d_model, batch=batch, seq=seq
    )

    def build(shape_fn):
        es = ShardedES(R.STRATEGIES[strategy](), n=population, sigma=R.SIGMA, lr=R.LR,
                       mesh=mesh, how=how, shaping=shape_fn)
        return es, es.init(key, params)

    es, state = build(shaping.centered_ranks)
    es_bare, state_bare = build(shaping.none)

    @jax.jit
    def full(state):
        pert, scaled = es.ask(state)
        return es.tell(state, pert, es.apply(transformer_block.loss, scaled, pert)(data))

    @jax.jit
    def no_shape(state):
        pert, scaled = es_bare.ask(state)
        return es_bare.tell(
            state, pert, es_bare.apply(transformer_block.loss, scaled, pert)(data)
        )

    @jax.jit
    def evaluate(state):
        pert, scaled = es.ask(state)
        return es.apply(transformer_block.loss, scaled, pert)(data)

    f_full, f_eval = flops_of(full, state), flops_of(evaluate, state)

    if static:
        t_full = iqr = t_bare = t_eval = t_floor = float("nan")
    else:
        with jax.default_matmul_precision("highest"):
            t_full, iqr = timed(full, state, warmup, repeats)
            t_bare, _ = timed(no_shape, state_bare, warmup, repeats)
            t_eval, _ = timed(evaluate, state, warmup, repeats)
            t_floor = floor_seconds(mesh, warmup, repeats)

    return {
        "d_model": d_model, "population": population, "strategy": strategy,
        "how": how, "devices": devices,
        "full": t_full, "full_iqr": iqr,
        "no_shape": t_bare, "eval": t_eval, "floor": t_floor,
        "flops": f_full, "eval_flops": f_eval,
        "shaping": t_full - t_bare,
        "tell": t_full - t_eval,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=pathlib.Path, default=HERE / "sweep.yaml")
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None,
                    help="only this model size. The sweep's largest shape needs ~50 GB, so "
                         "on a smaller card this is what lets the rest of the grid run")
    ap.add_argument("--population", type=int, default=None,
                    help="override the population. The default is the largest in the config, "
                         "which is the slowest thing to compile; a smaller one is what you "
                         "want when the question is whether D>1 runs at all")
    ap.add_argument("--strategies", default="iid_gaussian,seed_regenerated",
                    help="comma separated; the default is the two extremes of the memory "
                         "range, since M1 found the efficiency barely depends on this axis")
    ap.add_argument("--static", action="store_true",
                    help="FLOPs only, no timings. Valid on simulated devices, and the "
                         "columns that answer the scaling question")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args(argv)

    import yaml  # noqa: PLC0415
    from feasible import populations  # noqa: PLC0415

    cfg = yaml.safe_load(args.config.read_text())
    batch, seq = int(cfg.get("batch", 8)), int(cfg.get("seq", 32))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    available = jax.device_count()

    print(f"{available} x {jax.devices()[0].device_kind} ({jax.devices()[0].platform}), "
          f"jax {jax.__version__}")
    if jax.devices()[0].platform == "cpu" and not args.static:
        print("WARNING: simulated devices share memory and never communicate. These timings "
              "are fiction; see docs/06. Pass --static for the columns that are not.")
    if args.static:
        print("\nFLOPs only (--static). The timing columns are omitted rather than faked, "
              "and the\nFLOPs columns are a property of the compiled program, so they hold "
              "on any backend.\n")
    else:
        print("\nmilliseconds per generation, median of "
              f"{args.repeats} after {args.warmup} warm-up\n")
    if args.static:
        print(f"{'d':>6}{'N':>6}  {'strategy':17}{'how':>4}{'D':>3}"
              f"{'evalGFLOP':>12}{'fullGFLOP':>12}")
    else:
        print(f"{'d':>6}{'N':>6}  {'strategy':17}{'how':>4}{'D':>3}"
              f"{'full':>9}{'+-':>7}{'eval':>9}{'tell':>9}{'shaping':>9}{'floor':>8}"
              f"{'evalGFLOP':>11}")

    rows = []
    for d_model in cfg["d_model"]:
        if args.d_model and d_model != args.d_model:
            continue
        # The largest population per model size: the noise floor and the memory ceiling both
        # bind there, and it is where a barrier would show up most clearly. Overridable
        # because that is also the slowest shape to compile, and a first run on unfamiliar
        # hardware wants to know whether D>1 works before it wants the hardest case.
        population = args.population or max(populations(cfg, "strong", d_model, 1))
        for strategy in strategies:
            for how in cfg["how"]:
                for devices in cfg["devices"]:
                    if devices > available or (args.devices and devices != args.devices):
                        continue
                    m = profile(d_model, population, strategy, how, devices,
                                batch, seq, args.warmup, args.repeats, args.static)
                    rows.append(m)
                    head = (f"{d_model:>6}{population:>6}  {strategy:17}"
                            f"{how:>4}{devices:>3}")
                    if args.static:
                        print(head + f"{m['eval_flops'] / 1e9:>12.1f}"
                                     f"{m['flops'] / 1e9:>12.1f}")
                    else:
                        print(head
                              + f"{m['full'] * 1e3:>9.2f}{m['full_iqr'] * 1e3:>7.2f}"
                              + f"{m['eval'] * 1e3:>9.2f}"
                              + f"{m['tell'] * 1e3:>9.2f}{m['shaping'] * 1e3:>9.2f}"
                              + f"{m['floor'] * 1e3:>8.2f}{m['eval_flops'] / 1e9:>11.1f}")

    print()
    by = {(r["d_model"], r["population"], r["strategy"], r["how"]): {} for r in rows}
    for r in rows:
        by[(r["d_model"], r["population"], r["strategy"], r["how"])][r["devices"]] = r
    print("how each part scales, D=1 -> D=max (a part that distributes should fall)")
    if args.static:
        print(f"  {'config':44}{'evalFLOP':>10}{'fullFLOP':>10}")
    else:
        print(f"  {'config':44}{'full':>8}{'eval':>8}{'tell':>8}{'shaping':>9}"
              f"{'floor':>8}{'evalFLOP':>10}")
    for k, v in sorted(by.items(), key=str):
        if len(v) < 2:
            continue
        lo, hi = min(v), max(v)
        def ratio(field):  # noqa: E306
            return v[hi][field] / v[lo][field] if v[lo][field] else float("nan")
        label = f"  d={k[0]} N={k[1]} {k[2]}/{k[3]} D{lo}->D{hi}".ljust(46)
        if args.static:
            print(label + f"{ratio('eval_flops'):>10.3f}{ratio('flops'):>10.3f}")
        else:
            print(label
                  + f"{ratio('full'):>8.2f}{ratio('eval'):>8.2f}{ratio('tell'):>8.2f}"
                  + f"{ratio('shaping'):>9.2f}{ratio('floor'):>8.2f}"
                  + f"{ratio('eval_flops'):>10.3f}")
    ideal = 1 / max((r["devices"] for r in rows), default=1)
    print(f"\n1.00 means the part did not change. Strong scaling wants eval near 1/D = "
          f"{ideal:.3f}.")

    flat = [k for k, v in by.items() if len(v) > 1
            and abs(v[max(v)]["eval_flops"] / v[min(v)]["eval_flops"] - 1.0) < 0.05]
    if flat:
        print(f"\n{len(flat)} of {len(by)} configurations do not distribute their evaluation "
              "at all:\nper-device FLOPs are unchanged from D=1 to D=max, so every device "
              "evaluates the whole\npopulation and wall clock cannot fall. Parallel "
              f"efficiency is then exactly 1/D ({ideal:.3f}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
