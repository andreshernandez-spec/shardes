#!/usr/bin/env python
"""E10's overlap caveat, closed in context: full generations, shaping on vs off.

    python barrier_context.py            # writes results-barrier-context/*.json

barrier.py measured the shaping barrier in isolation and its README says so: an
isolation number is the barrier's worst case, because a real generation may
overlap part of the latency behind compute. This runs the check the caveat asks
for: identical full generations (lowrank_r1, contraction B, D=8) with
shaping=none and shaping=centered_ranks, at the two largest cells the sweep
measured on this platform. The isolation numbers predict deltas of 1--2% of a
generation at these shapes; if the measured delta is within the repeat noise,
the honest sentence is "invisible in context at measured scales, as the
isolation ceiling predicts", and the 12 ms at N=2^18 remains the ceiling for
scales where a full generation does not fit this hardware anyway.

7 repeats rather than run.py's 5, because the effect being bounded is small.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402

from shardes import shaping as shapings  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402

CELLS = [(512, 1024), (2048, 256)]  # (d_model, N): the sweep's largest per d
WARMUP, REPEATS = 3, 7
OUT = HERE / "results-barrier-context"


def measure(d_model: int, n: int, shaping_name: str) -> dict:
    mesh = sharding.make_mesh(8)
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=d_model)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=d_model, batch=8, seq=32)
    es = ShardedES(LowRank(r=1), n=n, sigma=0.01, lr=0.05, mesh=mesh, how="B",
                   shaping=shapings.BY_NAME[shaping_name])
    state = es.init(key, params)

    @jax.jit
    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness)

    for _ in range(WARMUP):
        state = generation(state)
    jax.block_until_ready(state.params)
    seconds = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        state = generation(state)
        jax.block_until_ready(state.params)
        seconds.append(time.perf_counter() - t0)
    return {"config": {"d_model": d_model, "population": n, "devices": 8,
                       "strategy": "lowrank_r1", "how": "B",
                       "shaping": shaping_name},
            "seconds_all": seconds,
            "seconds_median": statistics.median(seconds)}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    env = harness.capture_env(HERE, ("results-barrier-context",))
    for d, n in CELLS:
        pair = {}
        for s in ("none", "centered_ranks"):
            rec = measure(d, n, s)
            rec["env"] = env
            (OUT / f"d={d}__N={n}__s={s}.json").write_text(json.dumps(rec, indent=1))
            pair[s] = rec["seconds_median"]
            print(f"d={d} N={n} shaping={s}: {pair[s] * 1e3:.2f} ms", flush=True)
        delta = pair["centered_ranks"] - pair["none"]
        print(f"d={d} N={n}: in-context shaping delta {delta * 1e6:.0f} us "
              f"({100 * delta / pair['none']:.2f}% of a generation)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
