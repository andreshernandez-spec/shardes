#!/usr/bin/env python
"""Generate the CPU-simulated reference that the real-hardware check compares against.

    python reference.py            # writes reference.json
    python reference.py --check    # non-zero exit if the committed file is stale

Gate G1 criterion 2 asks that `test_device_invariance` pass on CPU-8 *and* be reproduced on
real GPUs. This file is the first half: one generation of ask/eval/tell at a fixed seed on
`--devices` simulated CPU devices, with the resulting params written out.

`tests/gpu/test_device_invariance_gpu.py` is the second half, and it needs a committed
artifact rather than a live CPU run: the point is to check the *simulated-device shortcut*,
so the reference has to be produced by the shortcut and then travel to hardware that does not
share its assumptions.

Deliberately small. This is a correctness artifact, and it has to run in seconds inside a
Kaggle session whose GPU quota is the scarce thing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from shardes import sharding
from shardes.core import ShardedES
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

HERE = pathlib.Path(__file__).resolve().parent
REFERENCE = HERE / "reference.json"

#: Fixed, and part of the artifact. Changing any of these invalidates a committed reference,
#: which is why they are here rather than on the command line.
SEED, N, SIGMA, LR = 0, 32, 0.01, 0.05

STRATEGIES = {
    "iid_gaussian": IIDGaussian,
    "seed_regenerated": SeedRegenerated,
    "mirrored_lr1": lambda: Mirrored(LowRank(r=1)),
    # Unmirrored, and it was missing until 2026-08-03. `mirrored_lr1` wraps it, so the pair
    # looked like coverage of the low-rank path while the only strategy the phase 2 sweep
    # ever failed on was the one no invariance test ran. Mirroring is a sign flip over an
    # inner perturbation, not a different code path, so it does not stand in for this.
    "lowrank_r1": lambda: LowRank(r=1),
}


def params0():
    k1, k2 = jax.random.split(jax.random.key(SEED))
    return {
        "w": jax.random.normal(k1, (6, 4), jnp.float32),
        "b": jax.random.normal(k2, (4,), jnp.float32),
    }


def sphere(p, _x):
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p))


def one_generation(strategy_name: str, n_devices: int, how: str) -> np.ndarray:
    """One ask/eval/tell, returned as a flat float64 vector for comparison only.

    Flattened *here*, on the host, and never inside `src/` — invariant 1 is about the
    library, and a reference file needs one array rather than a tree.
    """
    mesh = sharding.make_mesh(n_devices)
    es = ShardedES(STRATEGIES[strategy_name](), n=N, sigma=SIGMA, lr=LR, mesh=mesh, how=how)
    state = es.init(jax.random.key(SEED), params0())

    @jax.jit
    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(sphere, state, pert)(jnp.zeros(()))
        return es.tell(state, pert, fitness)

    out = generation(state)
    return np.concatenate(
        [np.asarray(x, dtype=np.float64).ravel() for x in jax.tree.leaves(out.params)]
    )


def build(n_devices: int) -> dict:
    devices = jax.devices()
    return {
        "config": {"seed": SEED, "n": N, "sigma": SIGMA, "lr": LR,
                   "n_devices": n_devices, "model": "sphere", "params": "w(6,4) b(4,)"},
        "env": {"jax": jax.__version__, "platform": devices[0].platform,
                "device_kind": getattr(devices[0], "device_kind", "unknown"),
                "device_count": len(devices)},
        "updates": {
            f"{name}/{how}": one_generation(name, n_devices, how).tolist()
            for name in STRATEGIES
            for how in ("A", "B")
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if jax.devices()[0].platform != "cpu":
        print("the reference must be generated on CPU: it is the simulated-device result "
              "that hardware is checked against. Set JAX_PLATFORMS=cpu.", file=sys.stderr)
        return 1

    built = build(args.devices)
    if args.check:
        if not REFERENCE.exists():
            print("reference.json missing; run reference.py", file=sys.stderr)
            return 1
        old = json.loads(REFERENCE.read_text())
        drift = {
            k: float(np.max(np.abs(np.array(v) - np.array(old["updates"][k]))))
            for k, v in built["updates"].items()
            if k in old["updates"]
        }
        worst = max(drift.values(), default=0.0)
        if set(drift) != set(built["updates"]) or worst > 0:
            print(f"reference.json is stale (worst drift {worst:.2e}); re-run", file=sys.stderr)
            return 1
        print(f"reference.json current ({len(drift)} entries, exact)")
        return 0

    REFERENCE.write_text(json.dumps(built, indent=1))
    print(f"wrote {REFERENCE}")
    print(f"  {len(built['updates'])} entries, {args.devices} simulated CPU devices, "
          f"jax {built['env']['jax']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
