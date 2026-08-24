#!/usr/bin/env python
"""What the contraction itself costs, split from the collective it carries.

    python contraction_isolation.py                 # all local devices, the sweep's cells
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      JAX_PLATFORMS=cpu python contraction_isolation.py --smoke

`allreduce_ladder.py` times the two collectives in isolation and `run.py` times the whole
generation. The term between them is the contraction: how much arithmetic strategy B splits
D ways and strategy A repeats on every device. Without it the crossover has no cost model,
only two endpoints, and the placement claim in the paper is a direction rather than a
prediction.

Three slopes per cell, all with the ladder's discipline (warm-up, fenced repeats, median,
and a dependent chain so nothing is hoisted or dispatched once and reused):

  A        `contract_replicated` with members-sharded weights: the full contraction on
           every device plus the 4N fitness all-gather it needs first.
  B        `contract_sharded`: the local N/D contraction plus the model-sized all-reduce.
  B_local  the same shard_map body with the psum removed: the local contraction alone.

From those, three numbers the paper does not otherwise have:

  C        = A slope, the replicated contraction (the 4N gather is ~8 us, ladder)
  C/D      = B_local slope, which is a check that the contraction really does shard
  ar       = B - B_local, the all-reduce AS THE ES LOOP PAYS IT, inside a program that is
             already running and holding a model's worth of live buffers. The ladder's
             number is the same collective in an otherwise empty program; the gap between
             them is the part of the cost model that cannot be predicted from bytes.

The prediction the sweep then tests is one line:

    t_A - t_B = C (D-1)/D + ag(4N) - ar

`timemodel.py` reads these records plus the ladder and checks it cell by cell.

Chain, not one-shot. Each variant is timed at K in {1, 9} dependent contractions under a
`lax.scan`, and the slope is the per-contraction cost; the k=1 intercept is dispatch, which
on the A100 is 0.32 ms and would swamp a rank-1 contraction. The scan also bounds memory:
unrolled, nine live copies of a d=2048 i.i.d. population do not fit. Weights are bumped by a
carry-derived scalar every step so the all-gather cannot be hoisted out of the loop; the
member-id gather is loop-invariant and is hoisted, which costs A one 4N int32 gather across
the whole chain instead of nine (~8 us total, inside the noise).

One JSON per cell under results-contraction/, stamped with the environment. A cell whose
file exists is skipped, so a session that dies partway resumes over what it wrote, and
`--budget` stops cleanly between cells and exits 2 rather than being killed inside one.
Cells run cheapest first (low rank before dense, small before large) so a short budget
buys the crossover's own side of the grid rather than a single dense cell.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import harness  # noqa: E402
from run import STRATEGIES  # noqa: E402
from shardes import contraction, sharding  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.sharding import POP  # noqa: E402

OUTPUTS = ("results-contraction",)

#: The F2 grid: the (d, N) cells run.py swept, at the populations each model size fits.
CELLS = {512: (256, 1024), 2048: (128, 256)}
CHAIN = (1, 9)
#: cheapest first: the low-rank arms are where the crossover's open term is, and they
#: compile and run in a fraction of what a dense or regenerating cell costs.
ORDER = ("lowrank_r1", "mirrored_lr1", "mirrored_seed", "iid_gaussian", "seed_regenerated")
SEED = 0
BUMP = 1e-12  # keeps the chain dependent without moving the numbers


def timed(fn, args, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    s = sorted(ts)
    return {"median": statistics.median(ts), "all": ts,
            "iqr": [s[len(s) // 4], s[-1 - len(s) // 4]]}


def _scalar(tree) -> jnp.ndarray:
    """One number the whole carry depends on, so the chain cannot be reordered away."""
    return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(tree)).astype(jnp.float32)


def make_a(strategy, base_key, params, mesh, k: int):
    """Strategy A, k dependent replicated contractions. Plain jit: A is not sharded."""

    def prog(p0, ids, weights):
        def step(carry, _):
            w = weights + BUMP * _scalar(carry)
            update = contraction.contract_replicated(
                strategy, base_key, carry, ids, w, mesh)
            return jax.tree.map(lambda a, u: a + BUMP * u, carry, update), None

        carry, _ = jax.lax.scan(step, p0, None, length=k)
        return _scalar(carry)

    return jax.jit(prog)


def make_b(strategy, base_key, params, mesh, k: int, comm: bool):
    """Strategy B's shard_map body, k dependent local contractions, psum on or off.

    `comm=False` is `contract_sharded` with the one line removed. The output is a (1,)
    per-device array under `out_specs=P(POP)` in both cases, so the two programs differ by
    the psum and by nothing else: with `out_specs=P()` the no-psum variant does not type,
    because nothing tells shard_map the partial is replicated.
    """

    def local(ids_shard, w_shard):
        c0 = jax.tree.map(lambda x: jax.lax.pcast(x, (POP,), to="varying"), params)

        def step(carry, _):
            w = w_shard + BUMP * _scalar(carry)
            pert = strategy.sample(base_key, carry, ids_shard)
            partial = strategy.contract(pert, w)
            if comm:
                partial = jax.tree.map(lambda leaf: jax.lax.psum(leaf, POP), partial)
            return jax.tree.map(lambda a, u: a + BUMP * u, carry, partial), None

        carry, _ = jax.lax.scan(step, c0, None, length=k)
        return _scalar(carry)[None]

    return jax.jit(shard_map(local, mesh=mesh, in_specs=(P(POP), P(POP)),
                             out_specs=P(POP), check_vma=False))


def slope(runs: dict) -> float:
    k0, k1 = CHAIN
    return (runs[str(k1)]["median"] - runs[str(k0)]["median"]) / (k1 - k0)


def measure(strategy_name, d_model, n, mesh, d_count, warmup, repeats, precision):
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=d_model)
    strategy = STRATEGIES[strategy_name]()
    member = NamedSharding(mesh, P(POP))
    rep = sharding.replicated(mesh)

    ids = jax.device_put(jnp.arange(n, dtype=jnp.int32), member)
    weights = jax.device_put(jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32), member)
    p0 = jax.device_put(params, rep)

    out = {}
    with jax.default_matmul_precision(precision):
        for name, build in (
            ("A", lambda k: (make_a(strategy, key, params, mesh, k), (p0, ids, weights))),
            ("B", lambda k: (make_b(strategy, key, params, mesh, k, True), (ids, weights))),
            ("B_local", lambda k: (make_b(strategy, key, params, mesh, k, False),
                                   (ids, weights))),
        ):
            runs = {}
            for k in CHAIN:
                fn, fnargs = build(k)
                runs[str(k)] = timed(fn, fnargs, warmup, repeats)
            out[name] = {"chain": runs, "slope_seconds": slope(runs)}

    c_full = out["A"]["slope_seconds"]
    c_local = out["B_local"]["slope_seconds"]
    out["contraction_seconds"] = c_full
    out["contraction_local_seconds"] = c_local
    out["allreduce_insitu_seconds"] = out["B"]["slope_seconds"] - c_local
    # 1.0 means the contraction shards perfectly; above 1.0 is what A repeats for nothing.
    out["shard_ratio"] = (c_full / (d_count * c_local)) if c_local > 0 else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="d=64, N=16, one strategy")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--precision", default="highest", help="matches the F2 sweep")
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds; stop between cells and exit 2 when spent")
    args = ap.parse_args(argv)
    started = time.perf_counter()

    devs = jax.devices()
    d_count = len(devs)
    mesh = sharding.make_mesh(d_count)
    kind = devs[0].device_kind.replace(" ", "-").lower()
    out_dir = HERE / "results-contraction"
    out_dir.mkdir(exist_ok=True)
    env = harness.capture_env(HERE, OUTPUTS)

    cells = {64: (16,)} if args.smoke else CELLS
    names = args.strategies or (["lowrank_r1"] if args.smoke else list(ORDER))
    names = [n for n in ORDER if n in names] + [n for n in names if n not in ORDER]
    stopped = False

    for name in names:
        for d_model, pops in sorted(cells.items()):
            for n in sorted(pops):
                if args.budget and time.perf_counter() - started > args.budget:
                    print(f"STOPPED: budget {args.budget:.0f}s spent before "
                          f"d={d_model} N={n} s={name}", flush=True)
                    stopped = True
                    break
                slug = f"d={d_model}__N={n}__s={name}__{kind}__D{d_count}"
                path = out_dir / f"{slug}.json"
                if path.exists() and not args.smoke:
                    print(f"skip {slug} (exists)", flush=True)
                    continue
                try:
                    rec = measure(name, d_model, n, mesh, d_count,
                                  args.warmup, args.repeats, args.precision)
                except Exception as exc:  # a cell that does not fit is a result
                    rec = {"failed": f"{type(exc).__name__}: {exc}"}
                    print(f"FAIL {slug}: {type(exc).__name__}: {exc}", flush=True)
                rec.update({"config": {"d_model": d_model, "population": n,
                                       "strategy": name, "devices": d_count},
                            "device_kind": devs[0].device_kind,
                            "platform": devs[0].platform,
                            "matmul_precision": args.precision,
                            "chain": list(CHAIN), "env": env})
                path.write_text(json.dumps(rec, indent=1))
                if "failed" not in rec:
                    print(f"{slug}: C {rec['contraction_seconds'] * 1e3:8.3f} ms, "
                          f"C/D {rec['contraction_local_seconds'] * 1e3:8.3f} ms, "
                          f"allreduce {rec['allreduce_insitu_seconds'] * 1e6:8.1f} us, "
                          f"shard {rec['shard_ratio'] or float('nan'):.2f}", flush=True)
            if stopped:
                break
        if stopped:
            break
    print(f"wrote {out_dir}")
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
