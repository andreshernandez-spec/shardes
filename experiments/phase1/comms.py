#!/usr/bin/env python
"""Communication accounting: analytic prediction next to instrumented measurement.

    python comms.py                # the table
    python comms.py --check        # non-zero exit if any row disagrees

docs/02-phase1-sharded-core.md, showcase artifact 3: "bytes moved per generation, per
strategy, as a function of N, d, D — analytic prediction next to instrumented measurement.
If they disagree, you've found a bug, which is the point."

The two strategies (docs/02 C1.3):

    A  scalar all-reduce, replicated regeneration.  Gathers the N shaped weights *and* the
       N member ids, then every device regenerates all N perturbations and contracts.
           predicted:  two all-gathers of N 4-byte values  =  8N bytes off each device
    B  model-size all-reduce of the partial update. Each device contracts its own members
       into a params-shaped partial, then psums.
           predicted:  one all-reduce of |params| float32  =  4d bytes

**Measured from compiled HLO, not from a profiler.** Every collective in the optimized
module is read back with its shape, so the numbers are what XLA will actually run rather
than what this file believes it asked for. That is the only version of this table worth
printing: an analytic prediction checked against another analytic prediction proves nothing.

Runs on simulated CPU devices, which model *bytes* faithfully and interconnect not at all.
Every number here is a volume; none is a time. Phase 2 measures the time.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys

import jax
import jax.numpy as jnp

from shardes import contraction, sharding
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

DTYPE_BYTES = 4  # f32, and s32 for the member ids

#: bytes per element, by HLO element-type name.
WIDTH = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "s16": 2, "u16": 2,
         "f32": 4, "s32": 4, "u32": 4, "f64": 8, "s64": 8, "u64": 8}

#: `%name = <shape> <op>(...)`, where <shape> is one array or a tuple of them. The tuple
#: case is not a curiosity: XLA merges the per-leaf psums of a params-shaped update into a
#: single tuple-valued all-reduce, so a parser that reads only the first element reports
#: one leaf and silently omits the model. That is how this file first measured Strategy B
#: at 64 B against a predicted 2112.
ASSIGN = re.compile(r"=\s*(.+?)\s+(all-gather|all-reduce|all-to-all|collective-permute)\(")
ARRAY = re.compile(r"([a-z]+\d*)\[([0-9,]*)\]")


def collectives(fn, *args) -> list[tuple[str, int]]:
    """(kind, bytes) for every collective in the *optimized* HLO.

    Optimized rather than lowered, so a collective XLA fused away is not counted and one it
    introduced is. Every array in the result shape is summed, tuples included.
    """
    text = jax.jit(fn).lower(*args).compile().as_text()
    out = []
    for line in text.splitlines():
        m = ASSIGN.search(line)
        if not m:
            continue
        payload = 0
        for dtype, dims in ARRAY.findall(m.group(1)):
            if dtype not in WIDTH:
                continue
            n = 1
            for d in (int(x) for x in dims.split(",") if x):
                n *= d
            payload += WIDTH[dtype] * n
        out.append((m.group(2), payload))
    return out


def measure(strategy, n: int, d_devices: int, params) -> dict:
    mesh = sharding.make_mesh(d_devices)
    ids = sharding.member_ids(n, mesh)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(2), (n,), jnp.float32), sharding.members(mesh)
    )
    key = jax.random.key(3)

    got = {}
    for how, fn in (("A", contraction.contract_replicated),
                    ("B", contraction.contract_sharded)):
        ops = collectives(lambda i, w: fn(strategy, key, params, i, w, mesh), ids, weights)
        got[how] = sum(payload for _kind, payload in ops)
        got[how + "_ops"] = sorted({kind for kind, _ in ops})
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a measured volume contradicts the prediction")
    args = ap.parse_args(argv)

    strategies = {
        "iid_gaussian": IIDGaussian(),
        "seed_regenerated": SeedRegenerated(),
        "mirrored_lr1": Mirrored(LowRank(r=1)),
    }
    populations = (16, 64, 256)
    devices = (2, 4, 8)
    params = {
        "w": jax.random.normal(jax.random.key(0), (32, 16), jnp.float32),
        "b": jax.random.normal(jax.random.key(1), (16,), jnp.float32),
    }
    d_params = sum(int(jnp.size(x)) for x in jax.tree.leaves(params))

    print(f"params: d = {d_params} floats = {DTYPE_BYTES * d_params} B\n")
    print(f"{'strategy':<18}{'N':>5}{'D':>3}  "
          f"{'A pred':>9}{'A meas':>9}  {'B pred':>9}{'B meas':>9}   verdict")
    print("-" * 76)

    disagreements = []
    for name, strategy in strategies.items():
        for n, d in itertools.product(populations, devices):
            got = measure(strategy, n, d, params)
            # A gathers N scalars; B all-reduces the model. Both are the *payload*; XLA is
            # free to implement a D-way all-reduce as more than one message, so the check is
            # that the volume scales with the right quantity, not that it equals a constant.
            # A gathers the shaped weights *and* the member ids: `contract_replicated`
            # constrains both to replicated, because every device has to know which members
            # it is regenerating as well as how to weight them. The first version of this
            # prediction counted only the weights and was wrong by exactly a factor of two,
            # which the measurement caught.
            pred_a, pred_b = 2 * DTYPE_BYTES * n, DTYPE_BYTES * d_params
            # Both the kind and the volume, exactly. The kind is the claim -- A must never
            # all-reduce a model-sized array, B must never gather the population -- and the
            # byte count is what turns "O(N) vs O(d)" from a shape argument into a number.
            # Exact equality holds on every row, so anything looser would be hiding a
            # discrepancy rather than tolerating one.
            ok_a = "all-reduce" not in got["A_ops"] and got["A"] == pred_a
            ok_b = "all-gather" not in got["B_ops"] and got["B"] == pred_b
            verdict = "ok" if (ok_a and ok_b) else "MISMATCH"
            if verdict != "ok":
                disagreements.append((name, n, d, got))
            print(f"{name:<18}{n:>5}{d:>3}  {pred_a:>9}{got['A']:>9}  "
                  f"{pred_b:>9}{got['B']:>9}   {verdict}")

    crossover = d_params / 2
    print(f"""
A moves 8N bytes and does not depend on D or on the model. B moves 4d bytes and does not
depend on N. So A wins below N = d/2 = {crossover:.0f} members here, and B above it, and the
crossover moves with the model rather than with the device count. That is the phase diagram
docs/05 calls the money figure, in miniature and at one model size.

The claim this table exists to keep honest (docs/02 C1.3): "ES only all-reduces scalar
fitnesses" is true for A and false for B. Both are legitimate. Nothing public says otherwise
until Phase 2 measures where the crossover actually falls in wall-clock, which depends on
interconnect and is not answerable on simulated devices.""")

    if disagreements:
        print(f"\n{len(disagreements)} row(s) disagree with the prediction.", file=sys.stderr)
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
