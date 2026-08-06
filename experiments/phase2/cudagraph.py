#!/usr/bin/env python
"""Does this node need `--xla_gpu_enable_command_buffer=`, and what does it cost?

    python cudagraph.py                 # one generation per device count, plus step timing

The preflight to run on a rented multi-GPU node **before** the sweep, because the failure it
looks for is expensive and quiet.

**What it checks.** XLA:GPU builds its command buffers as CUDA graphs. On some multi-GPU
nodes the capture fails outright and every sharded configuration dies with:

    INTERNAL: CUDA error: Failed to add memset node to a CUDA graph:
    CUDA_ERROR_INVALID_VALUE [executable_name='jit_generation']

Measured on 2x A100-SXM4-80GB, driver 550.127.05, jax 0.11.0, 2026-08-06: all 16 `D=2`
rehearsal configurations failed and all `D=1` configurations passed. Setting
`--xla_gpu_enable_command_buffer=` to the empty value turns command buffers off and fixes it.
2x T4 passed without the flag on 2026-08-01, so this is a property of the node rather than of
CUDA, which is the reason to measure it rather than assume it.

**Why it is worth a preflight.** `run.py` records a failed configuration and continues, so a
sweep on 8 GPUs finishes, exits 0, and writes 64 single-device results next to 192 errors.
That reads as a completed sweep in every summary except the one that counts the errors.
`run.py` now refuses to start without the flag on a multi-GPU node; this script is how the
refusal was justified and how the cost was measured.

**The cost.** Command buffers cut launch overhead, so disabling them is not free in
principle. Measured at `d=512, N=256, strategy A`: 13.00 ms -> 13.03 ms per step on one
A100, which is 0.2% and inside the run-to-run spread. Re-measure here rather than trusting
that number on different hardware.

Run it twice to see both sides, since XLA reads `XLA_FLAGS` once at backend init:

    XLA_FLAGS="--xla_gpu_deterministic_ops=true" python cudagraph.py
    XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_enable_command_buffer=" python cudagraph.py

Exits 1 if any device count fails, so it can gate a session.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax  # noqa: E402

import run as R  # noqa: E402  SEED, SIGMA, LR and STRATEGIES, so they cannot drift

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402


def one_generation(devices: int, d_model: int, population: int, strategy: str,
                   batch: int = 8, seq: int = 32, steps: int = 8) -> dict:
    """Compile and step the same `generation` the sweep compiles, on `devices` devices.

    Strategy A and a single fixed seed: this is asking whether the compiled program runs at
    all, not what it computes, and A is the path that regenerates on every device.
    """
    key = jax.random.key(R.SEED)
    params = transformer_block.init(key, d_model=d_model)
    data = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=d_model, batch=batch, seq=seq
    )
    es = ShardedES(R.STRATEGIES[strategy](), n=population, sigma=R.SIGMA, lr=R.LR,
                   mesh=sharding.make_mesh(devices), how="A")
    state = es.init(key, params)

    @jax.jit
    def generation(state):
        pert, scaled = es.ask(state)
        return es.tell(state, pert, es.apply(transformer_block.loss, scaled, pert)(data))

    t0 = time.perf_counter()
    state = jax.block_until_ready(generation(state))
    compile_s = time.perf_counter() - t0

    # Warm-up discarded for the same reason run.py discards it: the first call is compilation.
    for _ in range(3):
        state = jax.block_until_ready(generation(state))
    took = []
    for _ in range(steps):
        t = time.perf_counter()
        state = jax.block_until_ready(generation(state))
        took.append(time.perf_counter() - t)
    return {"compile_s": compile_s, "step_ms": 1000 * statistics.median(took)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--strategy", default="iid_gaussian")
    args = ap.parse_args(argv)

    flags = os.environ.get("XLA_FLAGS", "")
    available = jax.device_count()
    print(f"{available} x {getattr(jax.devices()[0], 'device_kind', '?')}, jax {jax.__version__}")
    print(f"XLA_FLAGS={flags or '(unset)'}")
    print(f"command buffers: {'off' if R._COMMAND_BUFFER_OFF.search(flags) else 'ON'}\n")
    print(f"d={args.d_model} N={args.population} {args.strategy} how=A")
    print(f"{'devices':>8}{'compile s':>12}{'step ms':>10}  result")

    # Every power of two up to the node, because the failure is about sharding rather than
    # about a particular width, and D=1 is the control that says the node works at all.
    counts, d = [], 1
    while d <= available:
        counts.append(d)
        d *= 2

    failed = []
    for devices in counts:
        try:
            m = one_generation(devices, args.d_model, args.population, args.strategy)
            print(f"{devices:>8}{m['compile_s']:>12.1f}{m['step_ms']:>10.2f}  ok")
        except Exception as exc:  # noqa: BLE001  the point is to report it, not to handle it
            failed.append(devices)
            first = str(exc).strip().splitlines()[0][:110]
            print(f"{devices:>8}{'-':>12}{'-':>10}  FAILED: {first}")

    print()
    if failed:
        print(f"device counts {failed} cannot compile a generation on this node.")
        if not R._COMMAND_BUFFER_OFF.search(flags):
            print()
            print("`--xla_gpu_enable_command_buffer=` is not set. If the error above mentions")
            print("a CUDA graph, that is the fix, and run.py requires it here anyway:")
            print()
            print(f'    XLA_FLAGS="{R.DETERMINISM_FLAG} {R.COMMAND_BUFFER_FLAG}" \\')
            print("        python cudagraph.py")
        return 1
    print(f"every device count up to {available} compiles and steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
