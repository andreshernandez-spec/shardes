#!/usr/bin/env python
"""Does EGGROLL's reference hit the TPU rank-1 cliff shardes fixed? (It does.)

    git clone https://github.com/ESHyperscale/HyperscaleES && (cd HyperscaleES
      && git checkout b77f7d6 && pip install -e . --no-deps)   # never vendored
    JAX_PLATFORMS=cpu python probe_eggroll_tpu.py

No TPU needed: libtpu compiles for a v5e topology from any linux box
(probe_lr1.py explains the trick). This builds hyperscalees' own EggRoll
rank-1 update step, unmodified, via m4.py's loader, at the cell where
shardes' rank-1 path used to fail on a v5e and now runs (245 ms bf16 and
254 ms f32 measured, results-cost-tpu-v5e8, after the zero-column pad of
PR #54), and asks the TPU compiler for it.

Measured 2026-08-21 at their b77f7d6: RESOURCE_EXHAUSTED, 24.14 GiB of HLO
temporaries against 15.75 GiB of HBM. Same pathology class we diagnosed
(a contracting-dimension-1 dot strength-reduced into a multiply chain that
pins activation-sized f32 copies), in their graph, at a shape their paper's
population scale makes ordinary. The comparison is stated fairly: their step
carries their optimizer and pairing and runs f32 by default, so the byte
totals are not like-for-like against ours; the like-for-like fact is that
one reference compiles at this cell on this chip and the other does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec  # noqa: E402

import m4  # noqa: E402
from shardes import sharding  # noqa: E402

CELL = m4.Shape(512, 16384, 8, 32)


def main() -> int:
    topo = topologies.get_topology_desc(topology_name="v5e:2x4", platform="tpu")
    built = m4.arm_eggroll(CELL, sharding.make_mesh(1))
    why = m4._reason(built)
    if why:
        print(f"eggroll arm unavailable: {why}")
        return 2
    step, state0 = built

    tpu_mesh = Mesh(np.array(topo.devices[:1]), (sharding.POP,),
                    axis_types=(AxisType.Auto,))

    def aval(x):
        return jax.ShapeDtypeStruct(
            jax.numpy.shape(x), jax.numpy.result_type(x),
            sharding=NamedSharding(tpu_mesh, PartitionSpec()))

    astate = jax.tree.map(aval, state0)
    try:
        c = jax.jit(step.__wrapped__).lower(astate).compile()
        ma = c.memory_analysis()
        print(f"EGGROLL rank-1 d={CELL.d_model} N={CELL.population}: COMPILES "
              f"on a v5e chip, temp {ma.temp_size_in_bytes / 2**30:.2f} GiB "
              "(the cliff is gone upstream; update the paper's sentence)")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"EGGROLL rank-1 d={CELL.d_model} N={CELL.population} on a v5e "
              f"chip: {str(e).splitlines()[0]}")
        print("shardes at the same cell, post-pad: compiles at 14.25 GiB, "
              "measured 245 ms bf16 / 254 ms f32 (results-cost-tpu-v5e8)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
