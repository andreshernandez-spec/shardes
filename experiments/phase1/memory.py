#!/usr/bin/env python
"""Peak per-device perturbation storage vs device count. docs/02 showcase artifact 2.

    python memory.py                 # the table
    python memory.py --plot          # also figures/memory-per-device.png

The claim: per-device perturbation storage falls as 1/D while a replicated baseline stays
flat. What makes the plot worth printing is that the three strategies fall differently, and
two of them are flat for reasons that have nothing to do with sharding.

**Measured from the compiled executable, not predicted.** `compiled.memory_analysis()` reports
what XLA will allocate per device. A closed-form model of "n_local * |params| * 4" would
reproduce the docstrings rather than check them, which is the same trap `comms.py` fell into
and is the reason that file now agrees to the byte.

Simulated CPU devices model allocation faithfully and interconnect not at all. Confirm on
real GPUs before any of this goes in a paper (docs/02, traps).
"""

from __future__ import annotations

import argparse
import itertools
import sys

import jax
import jax.numpy as jnp

from shardes import contraction, sharding
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

MB = 1024 * 1024


def peak_bytes(strategy, n: int, d_devices: int, params, how: str = "B") -> int:
    """Peak per-device temporary allocation for one contraction.

    `temp_size_in_bytes` is the scratch XLA needs beyond its inputs and outputs, which is
    where a materialized `(n_local, *params)` perturbation shows up. Argument and output
    sizes are excluded on purpose: those are the model and the update, and both are
    replicated by design, so counting them would bury the term that actually scales with the
    population under a constant.
    """
    mesh = sharding.make_mesh(d_devices)
    ids = sharding.member_ids(n, mesh)
    weights = jax.device_put(
        jax.random.normal(jax.random.key(2), (n,), jnp.float32), sharding.members(mesh)
    )
    fn = contraction.BY_NAME[how]
    compiled = jax.jit(
        lambda i, w: fn(strategy, jax.random.key(3), params, i, w, mesh)
    ).lower(ids, weights).compile()
    analysis = compiled.memory_analysis()
    return int(getattr(analysis, "temp_size_in_bytes", 0) or 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args(argv)

    strategies = {
        "iid_gaussian": IIDGaussian(),
        "seed_regenerated": SeedRegenerated(),
        "mirrored_lr1": Mirrored(LowRank(r=1)),
    }
    params = {
        "w": jax.random.normal(jax.random.key(0), (128, 64), jnp.float32),
        "b": jax.random.normal(jax.random.key(1), (64,), jnp.float32),
    }
    d_params = sum(int(jnp.size(x)) for x in jax.tree.leaves(params))
    devices = (1, 2, 4, 8)

    print(f"N = {args.n} members, |params| = {d_params} floats = {4 * d_params / 1024:.1f} KiB")
    print("peak per-device scratch, Strategy B (each device contracts its own members)\n")
    print(f"{'strategy':<18}" + "".join(f"{'D=' + str(d):>12}" for d in devices) + "   1/D?")
    print("-" * 72)

    table = {}
    for name, strategy in strategies.items():
        row = [peak_bytes(strategy, args.n, d, params) for d in devices]
        table[name] = row
        # Does it actually halve? Compare D=1 against D=8 rather than eyeballing the row.
        ratio = row[0] / row[-1] if row[-1] else float("inf")
        verdict = f"{ratio:.1f}x" if row[-1] else "n/a"
        print(f"{name:<18}" + "".join(f"{b / MB:>11.2f}M" for b in row) + f"   {verdict}")

    # The comparison the table exists for, computed rather than eyeballed.
    iid_best = table["iid_gaussian"][-1]
    for name in ("mirrored_lr1", "seed_regenerated"):
        alone = table[name][0]
        if alone:
            print(f"\n  {name} on ONE device uses {iid_best / alone:.0f}x less than "
                  f"iid_gaussian on {devices[-1]}.")

    print("""
Read the rows, not the headline. Only one of these is a sharding result:

  iid_gaussian      materializes (n_local, *params), so its scratch is the population term
                    and it falls with D. This is the row the claim is about.
  seed_regenerated  flat, and flat at ~0. It never holds more than one member's noise
                    because `contract` is a scan, so there is no population term to divide.
                    Qiu's bet: pay compute to make this row a constant.
  mirrored_lr1      falls with D, but from a far smaller base: the factors are
                    n*r*(m+k) rather than n*m*k, and the product is never formed.

So "storage falls as 1/D" is true, exactly, and is the least interesting thing here. The
strategies differ by orders of magnitude at D=1, and **the choice of strategy dominates the
choice of device count**: the lines above quantify that, and it is the finding worth putting
in a README rather than the 1/D slope. Sharding divides; picking the right perturbation
scheme changes the exponent.

There is a second crossover in the plot that the table hides. mirrored_lr1 falls and
seed_regenerated does not, so they cross: below the crossing seed regeneration is the cheaper
way to hold a population, above it the factored perturbation is. That mirrors the comms
crossover in `comms.py` and has the same shape of answer -- which strategy wins depends on the
configuration, not on which paper you prefer.

A log y-axis is required, not cosmetic. On a linear one, both low-memory strategies are a
flat line at zero underneath iid_gaussian and the entire result is invisible.""")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: pip install -e '.[experiments]'", file=sys.stderr)
            return 1
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (name, row), marker in zip(table.items(), itertools.cycle("os^D")):
            ax.plot(devices, [max(b, 1) / MB for b in row], marker=marker, label=name, lw=1.6)
        ideal = [table["iid_gaussian"][0] / d / MB for d in devices]
        ax.plot(devices, ideal, "k:", lw=1, label="ideal 1/D from iid_gaussian")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("devices")
        ax.set_ylabel("peak per-device scratch (MiB)")
        ax.set_title(f"Perturbation storage per device, N = {args.n}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=9)
        out = __file__.rsplit("/", 1)[0] + "/figures/memory-per-device.png"
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
