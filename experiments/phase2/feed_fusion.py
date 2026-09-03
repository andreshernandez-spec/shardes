#!/usr/bin/env python
"""Does the replicated contraction cost more when its weights come from `apply`?

    python feed_fusion.py                      # d=512, N=256, the surviving cell
    python feed_fusion.py --d-model 512 --population 1024

One cell of the F2 grid keeps a residual the cost model does not account for:
`iid_gaussian` at d=512, N=256, where A is about 1.5 ms slower in a real
generation than `contraction_isolation.py` says the contraction costs. Holding
a population-sized buffer resident does not reproduce it
(`results-recheck-2026-09-03/`), so memory pressure is out. The candidate left
is the one the isolation README raised first: the chain feeds the contraction
from a scan carry, while a generation feeds it from the evaluation, and XLA
may fuse the two differently.

**This runs on one GPU, and that is faithful for the arm in question.** Under
placement A every device contracts the whole population, so A's local
contraction is the same computation at D=1 as at D=8; only B's shrinks with D.
The residual is an A-side residual, so a single device measures the right
thing. What a single device cannot see is anything that needs the collective,
which is what makes a negative result here informative rather than final.

Two programs that do exactly the same work and differ in one edge:

    tell_fed     ask, apply, tell   -- tell's weights are apply's output
    tell_cut     ask, apply, tell   -- tell's weights are an argument, and
                                      apply's output is summed into the result
                                      so it is still computed

Both pay `ask` and both pay `apply`, so neither cancels out of the comparison
and neither can be dropped as dead code. The only difference is whether the
contraction's weights depend on the evaluation, which is the dependence the
isolation chain does not have. `tell_fed - tell_cut` is that dependence,
priced. A gap is the mechanism; no gap rules it out at D=1 and leaves only
effects that need the collective.

An earlier version compared `tell_fed - apply_only` against a `tell` that had
no `apply` in it. That is not a like-for-like comparison: the subtraction
cancels `ask`, which the standalone `tell` still pays, and `ask` materializes
the whole population for this strategy. It reported a 42% gap that was mostly
`ask`.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import harness  # noqa: E402
from run import LR, SEED, SIGMA, STRATEGIES  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402

OUTPUTS = ("results-feed-fusion",)


def timed(fn, args, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "all": ts,
            "min": min(ts), "max": max(ts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--strategy", default="iid_gaussian")
    ap.add_argument("--how", default="A")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=51)
    ap.add_argument("--precision", default="highest")
    args = ap.parse_args(argv)

    mesh = sharding.make_mesh(1)
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=args.d_model)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=args.d_model,
                                         batch=args.batch, seq=args.seq)
    es = ShardedES(STRATEGIES[args.strategy](), n=args.population, sigma=SIGMA,
                   lr=LR, mesh=mesh, how=args.how)
    state = es.init(key, params)

    @jax.jit
    def tell_fed(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness), jnp.float32(0.0)

    @jax.jit
    def tell_cut(state, weights):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        # The contraction reads `weights`, not `fitness`; summing the fitness into
        # a second output keeps the evaluation alive so both programs do it.
        return es.tell(state, pert, weights), jnp.sum(fitness)

    with jax.default_matmul_precision(args.precision):
        # The fitness tell_const is handed: the real one, so the shaping sees the
        # same ranks and the contraction the same weights as tell_fed.
        pert0, s0 = es.ask(state)
        fitness = jax.block_until_ready(
            es.apply(transformer_block.loss, s0, pert0)(batch))

        out = {
            "tell_fed": timed(tell_fed, (state,), args.warmup, args.repeats),
            "tell_cut": timed(tell_cut, (state, fitness),
                              args.warmup, args.repeats),
        }

    fed = out["tell_fed"]["median"]
    const = out["tell_cut"]["median"]
    out["gap_seconds"] = fed - const
    out["config"] = {"d_model": args.d_model, "population": args.population,
                     "strategy": args.strategy, "how": args.how,
                     "devices": 1, "batch": args.batch, "seq": args.seq,
                     "matmul_precision": args.precision,
                     "warmup": args.warmup, "repeats": args.repeats}
    out["env"] = harness.capture_env(HERE, OUTPUTS)

    for k in ("tell_fed", "tell_cut"):
        r = out[k]
        print(f"{k:10s} {r['median'] * 1e3:8.3f} ms   "
              f"[{r['min'] * 1e3:.3f}, {r['max'] * 1e3:.3f}]")
    print()
    print(f"gap (fed - cut): {(fed - const) * 1e3:+.3f} ms  "
          f"({100 * (fed - const) / const:+.2f}% of the generation)")

    d = HERE / OUTPUTS[0]
    d.mkdir(exist_ok=True)
    f = d / f"d={args.d_model}__N={args.population}__s={args.strategy}__how={args.how}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
