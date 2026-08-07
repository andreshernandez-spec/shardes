#!/usr/bin/env python
"""M5: bytes moved per generation, measured against the Phase 1 analytic prediction.

    python comms.py --config sweep.yaml            # the table
    python comms.py --config sweep.yaml --json out.json

`docs/02` C1.3 predicts the two contraction strategies cost different things to communicate:

- **Strategy A**: all-gather the `N` fitness scalars. `O(N)`, independent of the model.
- **Strategy B**: one params-sized all-reduce of the partial update. `O(d^2)` for a
  transformer block, independent of the population.

Nothing public claims "ES only all-reduces scalars" until that is measured, which is what
this is. `tests/test_contraction.py::test_comm_volume_*` already assert *which* collectives
appear; this measures *how many bytes* go through them, which is the part the crossover
argument actually rests on.

**What is counted, precisely.** Every collective instruction in the optimized HLO, valued at
the size of its output buffer. That is the payload: the array the program hands to the
collective. It is not wire bytes. A ring all-reduce moves roughly `2(D-1)/D` times the
payload on the physical links and an all-gather roughly `(D-1)/D`, so the wire figure is a
constant factor away and depends on the algorithm XLA picks. The payload is the part that is
a property of the design rather than of the interconnect, and it is what `docs/02`'s `O(N)`
against `O(d)` is a claim about.

**The shaping barrier is measured by difference.** `centered_ranks` needs a global sort over
all `N` fitnesses, so it is an all-gather plus a wait; `none` is not a barrier at all
(`src/shardes/shaping.py`). Compiling the same configuration both ways and subtracting
isolates what the shaping costs, without needing to attribute individual HLO instructions to
source lines.

**Simulated devices are legitimate here, unlike for timing.** `docs/06` is emphatic that a
wall-clock scaling curve from `--xla_force_host_platform_device_count` is not a scaling
curve, because simulated devices share memory and never communicate. This measures the
compiled program rather than its execution: the collectives are in the HLO whether or not
any wire carries them. The caveat that remains is that XLA:CPU and XLA:GPU can lower the
same program differently, so the table records which backend produced it and the GPU run is
the one to quote.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax  # noqa: E402

import run as R  # noqa: E402  SEED, SIGMA, LR, STRATEGIES, so they cannot drift

from shardes import sharding, shaping  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402

#: The collectives that move data between devices. `collective-permute` and `all-to-all` are
#: not expected from this code and are listed so that one appearing is visible rather than
#: silently uncounted.
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
               "collective-permute")

#: `%name = <shape> <opcode>(...)`, one instruction per line.
#:
#: **The shape is captured non-greedily because it can contain spaces.** A `psum` over a
#: pytree compiles to a single tuple-shaped collective,
#: `%all-reduce = (f32[32,32]{1,0}, f32[32,32]{1,0}, ...) all-reduce(...)`, and a `(\S+)`
#: shape pattern stops at the first comma and matches nothing. That read as "strategy B
#: performs no collective at all", which is the one answer this script exists to disprove.
_INSTRUCTION = re.compile(r"^\s*%?[\w.\-]+ = (.*?) ([a-z\-]+)\(", re.MULTILINE)

#: XLA:GPU splits collectives into an async pair, and **the `-done` is the one to count**:
#:
#:     %all-reduce-start = ((f32[1572864]{0}), f32[1572864]{0}) all-reduce-start(...)
#:     %all-reduce-done  =   f32[1572864]{0}   all-reduce-done(%all-reduce-start)
#:
#: The `-start` output is a nested tuple of *(operand, result)*, so summing its shapes counts
#: the buffer twice. That read as exactly 2.00x the prediction for every strategy B
#: configuration on GPU, and 1.50x for A, where the all-gather operand is `N/D` and the
#: result is `N`. Neither is a lowering difference; both were this parser. XLA:CPU emits the
#: un-suffixed opcode with only the result shape, which is why the same code agreed with the
#: prediction exactly on CPU and had to be checked on a GPU to find the bug.
_COUNTED = {c: c for c in COLLECTIVES} | {f"{c}-done": c for c in COLLECTIVES}
_SHAPE = re.compile(r"(f32|f16|bf16|f64|s32|s64|u32|u8|pred)\[([\d,]*)\]")

_WIDTH = {"f64": 8, "s64": 8, "f32": 4, "s32": 4, "u32": 4, "f16": 2, "bf16": 2,
          "u8": 1, "pred": 1}


def payload_bytes(shape_text: str) -> int:
    """Bytes in an HLO shape, summing the elements of a tuple.

    A scalar is `f32[]`, which is 4 bytes and not 0: `[]` means rank zero, not empty.
    """
    total = 0
    for dtype, dims in _SHAPE.findall(shape_text):
        n = 1
        for d in dims.split(","):
            if d.strip():
                n *= int(d)
        total += n * _WIDTH[dtype]
    return total


def collective_bytes(hlo: str) -> dict[str, int]:
    """Payload bytes per collective opcode in a compiled HLO module."""
    out: dict[str, int] = collections.defaultdict(int)
    for shape_text, opcode in _INSTRUCTION.findall(hlo):
        if opcode in _COUNTED:
            out[_COUNTED[opcode]] += payload_bytes(shape_text)
    return dict(out)


def compile_generation(d_model: int, population: int, strategy: str, how: str, devices: int,
                       shape_fn, batch: int = 8, seq: int = 32) -> str:
    """The compiled HLO for one generation, set up exactly as `run.measure` does."""
    key = jax.random.key(R.SEED)
    params = transformer_block.init(key, d_model=d_model)
    data = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=d_model, batch=batch, seq=seq
    )
    es = ShardedES(R.STRATEGIES[strategy](), n=population, sigma=R.SIGMA, lr=R.LR,
                   mesh=sharding.make_mesh(devices), how=how, shaping=shape_fn)
    state = es.init(key, params)

    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(data)
        return es.tell(state, pert, fitness)

    with jax.default_matmul_precision("highest"):
        return jax.jit(generation).lower(state).compile().as_text()


def params_bytes(d_model: int) -> int:
    """Model size in float32, the analytic prediction for strategy B."""
    params = transformer_block.init(jax.random.key(R.SEED), d_model=d_model)
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(params))


def predicted(how: str, population: int, d_model: int) -> int:
    """`docs/02` C1.3. A gathers N fitness scalars; B all-reduces one params-sized array."""
    return 4 * population if how == "A" else params_bytes(d_model)


def measure(d_model: int, population: int, strategy: str, how: str, devices: int) -> dict:
    full = collective_bytes(
        compile_generation(d_model, population, strategy, how, devices, shaping.centered_ranks)
    )
    bare = collective_bytes(
        compile_generation(d_model, population, strategy, how, devices, shaping.none)
    )
    total, unshaped = sum(full.values()), sum(bare.values())
    return {
        "d_model": d_model, "population": population, "strategy": strategy,
        "how": how, "devices": devices,
        "bytes": total,
        "by_op": full,
        "bytes_without_shaping": unshaped,
        # The shaping barrier's own contribution, by difference. Negative would mean the
        # rank transform *removed* a collective, which would be worth knowing about.
        "shaping_bytes": total - unshaped,
        "predicted": predicted(how, population, d_model),
    }


def configurations(cfg: dict) -> list[tuple]:
    """Strong-scaling shapes only: weak mode changes N with D, so its communication is not
    a function of one population and the comparison against the prediction would be muddled.
    """
    from feasible import populations  # noqa: PLC0415  handles both config shapes

    out = []
    for d_model in cfg["d_model"]:
        for devices in cfg["devices"]:
            # `devices` is passed because the helper scales weak-mode populations by it; for
            # strong mode it is ignored and the same list comes back every time.
            for population in populations(cfg, "strong", d_model, devices):
                for strategy in cfg["strategies"]:
                    for how in cfg["how"]:
                        out.append((d_model, population, strategy, how, devices))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=pathlib.Path, default=HERE / "sweep.yaml")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--devices", type=int, default=None,
                    help="only this device count; default is every count in the config")
    args = ap.parse_args(argv)

    import yaml  # noqa: PLC0415
    cfg = yaml.safe_load(args.config.read_text())
    available = jax.device_count()

    print(f"{args.config.name} on {available} x {jax.devices()[0].device_kind} "
          f"({jax.devices()[0].platform}), jax {jax.__version__}")
    print("payload bytes per generation, from the compiled HLO; not wire bytes\n")
    print(f"{'d':>6}{'N':>6}  {'strategy':17}{'how':>4}{'D':>3}"
          f"{'measured':>12}{'predicted':>12}{'ratio':>8}{'shaping':>10}  ops")

    rows, mismatched = [], []
    for d_model, population, strategy, how, devices in configurations(cfg):
        if devices > available:
            continue
        if args.devices and devices != args.devices:
            continue
        m = measure(d_model, population, strategy, how, devices)
        rows.append(m)
        ratio = m["bytes"] / m["predicted"] if m["predicted"] else math.inf
        ops = ",".join(sorted(m["by_op"])) or "none"
        # One device has nothing to communicate, so the prediction does not apply and a
        # ratio there is not a result. Strategy A emits no collective at all at D=1
        # (`tests/test_contraction.py::test_one_device_needs_no_collective_at_all`); a `0.00`
        # in that column would read as a failed prediction rather than as an absent one.
        shown = "     -  " if devices == 1 else f"{ratio:>8.2f}"
        print(f"{d_model:>6}{population:>6}  {strategy:17}{how:>4}{devices:>3}"
              f"{m['bytes']:>12,}{m['predicted']:>12,}{shown}"
              f"{m['shaping_bytes']:>10,}  {ops}")
        # D=1 has nothing to communicate, so the prediction does not apply to it.
        if devices > 1 and not (0.5 <= ratio <= 4.0):
            mismatched.append((m, ratio))

    print()
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"wrote {args.json}")

    multi = [r for r in rows if r["devices"] > 1]
    if multi:
        a = [r for r in multi if r["how"] == "A"]
        b = [r for r in multi if r["how"] == "B"]
        if a:
            print(f"A: {min(r['bytes'] for r in a):,} to {max(r['bytes'] for r in a):,} bytes")
        if b:
            print(f"B: {min(r['bytes'] for r in b):,} to {max(r['bytes'] for r in b):,} bytes")
        shaped = [r for r in multi if r["shaping_bytes"]]
        print(f"shaping barrier contributes bytes in {len(shaped)}/{len(multi)} configurations")

    if mismatched:
        print(f"\n{len(mismatched)} configuration(s) are more than 4x off the "
              f"`docs/02` prediction:")
        for m, ratio in mismatched[:8]:
            print(f"  d={m['d_model']} N={m['population']} {m['strategy']}/{m['how']} "
                  f"D={m['devices']}: {m['bytes']:,} measured vs {m['predicted']:,} "
                  f"predicted ({ratio:.2f}x)")
        print("\nThe prediction counts the collective the design calls for. A large ratio "
              "means\nthe compiled program moves data this analysis did not account for, "
              "which is a\nresult about the implementation rather than about the interconnect.")
        return 1
    print("\nevery configuration is within 4x of the docs/02 prediction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
