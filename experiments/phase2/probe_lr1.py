#!/usr/bin/env python
"""Why mirrored_lr1 OOMs on a v5e where lr4 and lr16 run (results-cost-tpu-v5e8).

    pip install "jax[tpu]==<your jax version>"   # libtpu compiles for TPU with no TPU
    JAX_PLATFORMS=cpu python probe_lr1.py

Needs no TPU and no session: libtpu's compiler runs anywhere x86-linux, and
`jax.experimental.topologies.get_topology_desc("v5e:2x4", "tpu")` gives
compile-only v5e devices to point `jit(...).lower(...).compile()` at. The OOM
in the sweep was a compile-time static allocation failure, so it reproduces
bit-for-bit this way: at d=512, N=16384, bf16, the r=1 compile throws
RESOURCE_EXHAUSTED (16.08 G of HLO temporaries against 15.75 G) while r=4
compiles and ran at 258 ms on the real chip.

What the halved shape (N=8192, fits, so the optimized HLO exists) shows:

    r=1: temp 10.19 GiB   f32[pairs,8,32,512] converts: 4   producers: 59
    r=2: temp  9.22 GiB   f32[pairs,8,32,512] converts: 2   producers: 36
    r=4: temp  7.44 GiB   f32[pairs,8,32,512] converts: 2   producers: 36

No (n_members, m, k) tensor exists in any of them, so invariant 3 holds on
TPU: the perturbation is never materialized, at any rank. The excess is
activations. r=2 and r=4 share one graph structure; r=1 alone grows a third
more activation-sized f32 temporaries. The mechanism consistent with that is
XLA's canonicalization of dots with a contracting dimension of size 1: at r=1
the correction `(x @ B) @ A.T` in `LowRankWeight.__call__` has k=1, the
compiler rewrites it from a dot into a broadcast-multiply-reduce, and on TPU
the rewritten chain neither stays bf16 nor fuses back into the surrounding
dot fusions, pinning full f32 activation copies. The GPU compiler makes the
same shapes monotonic in r (measured: 24.05 GiB at r=1 vs 32.03 at r=4 for
the N=16384 cell), which is why the A100 sweep shows no such anomaly.

The fix: `LowRankWeight._factors` pads r=1 to a rank-2 dot with a zero column
(bitwise-equal in bf16, 1-2 ulp in f32; `test_rank1_pad_is_numerically_
invisible` pins that). Hand-writing the multiply instead was measured first
and changed nothing (temp identical to the broken path), which is how the pad
earned its place: the multiply chain itself is what the TPU scheduler handles
badly, not who writes it. With the pad, the failing cell compiles at 14.25 G
and the halved shape drops from 10.19 to 7.18 GiB.

So this script is now the regression check: step 1 asserts the once-failing
cell COMPILES, and exits non-zero if the OOM ever comes back.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402

ACT = "4096,8,32,512"  # (mirrored pairs, batch, seq, d) at the halved probe shape


def compile_for_v5e(r: int, d_model: int, n: int):
    """The exact cost.py generation, compiled for one compile-only v5e chip."""
    topo = topologies.get_topology_desc(topology_name="v5e:2x4", platform="tpu")
    tpu_mesh = Mesh(np.array(topo.devices[:1]), (sharding.POP,),
                    axis_types=(AxisType.Auto,))
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=d_model)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=d_model, batch=8, seq=32)
    # init needs concrete arrays, so it runs on CPU; lower() only needs avals
    # carrying the TPU mesh. D=1, so a replicated spec is exact for every leaf.
    es_cpu = ShardedES(Mirrored(LowRank(r=r)), n=n, sigma=0.01, lr=0.05,
                       mesh=sharding.make_mesh(1), compute_dtype=jnp.bfloat16)
    state = es_cpu.init(key, params)
    astate = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape, x.dtype, sharding=NamedSharding(tpu_mesh, PartitionSpec())),
        state,
    )
    es = ShardedES(Mirrored(LowRank(r=r)), n=n, sigma=0.01, lr=0.05,
                   mesh=tpu_mesh, compute_dtype=jnp.bfloat16)

    def loss32(p, b):
        return transformer_block.loss(p, b).astype(jnp.float32)

    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(loss32, state, pert)(batch)
        return es.tell(state, pert, fitness)

    return jax.jit(generation).lower(astate).compile()


def main() -> int:
    print("1. The once-failing cell (d=512, N=16384, bf16), must compile:")
    try:
        c = compile_for_v5e(1, 512, 16384)
    except Exception as e:  # noqa: BLE001
        print(f"   REGRESSION, r=1 fails again: {str(e).splitlines()[0]}")
        return 1
    print(f"   r=1: compiles, temp "
          f"{c.memory_analysis().temp_size_in_bytes / 2**30:.2f} GiB "
          "(was RESOURCE_EXHAUSTED at 16.08 G before the pad)")
    c = compile_for_v5e(4, 512, 16384)
    print(f"   r=4: compiles, temp "
          f"{c.memory_analysis().temp_size_in_bytes / 2**30:.2f} GiB")

    print("\n2. Halved shape (N=8192), graph structure by rank:")
    for r in (1, 2, 4):
        c = compile_for_v5e(r, 512, 8192)
        txt = c.as_text()
        conv = len(re.findall(rf"f32\[{ACT}\][^\n]*convert", txt))
        prod = len(re.findall(rf"= f32\[{ACT}\][^\n]*(dot|fusion)", txt))
        mat = txt.count("8192,512,512") + txt.count("4096,512,512")
        print(f"   r={r}: temp {c.memory_analysis().temp_size_in_bytes / 2**30:5.2f} GiB"
              f"   f32-activation converts {conv}, producers {prod},"
              f"   materialized (N,m,k) tensors {mat}")
    print("\nExpected with the pad in: r=1 is no longer the structural outlier "
          "(before it held 10.19 GiB against r=4's 7.44); materialized count 0 at "
          "every rank (invariant 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
