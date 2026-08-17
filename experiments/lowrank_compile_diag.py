#!/usr/bin/env python
"""Split tell's first-call cost per rank: trace time, HLO size, XLA compile time, run time.

Written to diagnose the pilot's lr16 arm spending 6622 s and 6845 s in generations 0
and 1 (results/pilot-a100-2026-08-16). Findings on a laptop CPU, 12-layer model:

    before the fix   r=1: 63.7s   r=4: 187.7s   r=16: 974.9s   (hlo 6.3k/9.6k/21.9k lines)
    after the fix    r=1: 63.5s   r=4:  68.1s   r=16:  70.6s   (hlo flat, ~7.2k lines)

The cause: tell regenerates the factors from seeds (contraction re-runs sample inside
the jitted graph), and sample built each leaf's factor matrix with a Python loop over
the 2r column keys, unrolling 2r coupling subgraphs per rank-2 leaf into the trace.
The fix vmaps the coupling over the column-key array instead, which is bit-identical
(tests/test_strategies.py::test_lowrank_sample_column_seed_contract) and leaves the
graph size rank-independent (::test_lowrank_sample_graph_is_rank_independent)."""
import time

import jax
import jax.numpy as jnp

from shardes import sharding
from shardes.core import ShardedES
from shardes.problems import qwen2
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored

cfg = qwen2.Config(vocab=2048, d_model=192, n_layers=12, n_heads=6, n_kv_heads=2,
                   d_ff=512, rope_theta=1e4)
params = qwen2.init(jax.random.key(0), cfg, dtype=jnp.bfloat16)

for r in [1, 4, 16]:
    es = ShardedES(Mirrored(LowRank(r=r)), n=30, sigma=0.001, lr=5e-7,
                   mesh=sharding.make_mesh(None), compute_dtype=jnp.bfloat16)
    state = es.init(jax.random.key(1), params)
    pert, state = es.ask(state)
    fit = jnp.linspace(-1.0, 1.0, 30, dtype=jnp.float32)

    t = time.perf_counter()
    lowered = jax.jit(es.tell).lower(state, pert, fit)
    t_lower = time.perf_counter() - t

    hlo = lowered.compiler_ir("hlo").as_hlo_module().to_string()
    n_instr = hlo.count("\n")

    t = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t

    t = time.perf_counter()
    out = compiled(state, pert, fit)
    jax.block_until_ready(out.params)
    t_run = time.perf_counter() - t

    print(f"r={r:2d}  lower={t_lower:6.1f}s  hlo_lines={n_instr:8d}  "
          f"compile={t_compile:7.1f}s  run={t_run:5.1f}s", flush=True)
