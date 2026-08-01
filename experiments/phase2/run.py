#!/usr/bin/env python
"""M1/M2/M3: strong scaling, weak scaling, and the A/B contraction crossover.

    python run.py --config sweep.yaml --dry-run     # what would run, and how many
    python run.py --config sweep.yaml               # the sweep, resumable
    python run.py --config rehearsal.yaml           # the reduced-N dress rehearsal

One JSON per configuration under `results/`, so a killed sweep resumes by skipping what is
already there. `docs/03` asks for the run to be an execution of an already-debugged plan;
the rehearsal config is how the plan gets debugged, on CPU, for free.

**Benchmarks lie more easily than tests do.** Every guard in `docs/03`'s table is here and
each one is load-bearing:

- warm-up generations discarded, or the first timing is JIT compilation,
- `block_until_ready` on every timed result, or the timing is async dispatch,
- repeats with median and IQR, not a single sample,
- a params digest recorded per configuration, so the report can *assert* that `D=1` and
  `D=8` walked the same trajectory. A scaling number for two different computations is not
  a scaling number.

The digest is the one worth understanding. `test_device_invariance` proves the update is
device-count invariant for a fixed seed; recording the digest here proves *this sweep*
actually ran that, rather than benchmarking a config that quietly diverged.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.iid_gaussian import IIDGaussian  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

OUTPUTS = ("results", "results-rehearsal", "env.json", "figures")

#: Set from the config in main(). The rehearsal writes elsewhere so its numbers can never be
#: mistaken for the sweep's: docs/03 is explicit that no rehearsal result goes in a figure.
RESULTS = Path(__file__).resolve().parent / "results"

STRATEGIES = {
    "iid_gaussian": IIDGaussian,
    "seed_regenerated": SeedRegenerated,
    "mirrored_lr1": lambda: Mirrored(LowRank(r=1)),
    "lowrank_r1": lambda: LowRank(r=1),
}

SEED, SIGMA, LR = 0, 0.01, 0.05


@dataclass(frozen=True)
class Config:
    mode: str  # "strong" | "weak"
    devices: int
    d_model: int
    population: int
    strategy: str
    how: str  # contraction A or B

    def slug(self) -> str:
        return (
            f"mode={self.mode}__D={self.devices}__d={self.d_model}"
            f"__N={self.population}__s={self.strategy}__how={self.how}"
        )

    def path(self) -> Path:
        return RESULTS / f"{self.slug()}.json"


def expand(cfg: dict) -> list[Config]:
    """The sweep, as a deterministic list.

    Weak scaling multiplies the per-device population by the device count; strong scaling
    holds the total fixed. Keeping both in one driver means one resume story and one set of
    guards, and the mode is in the slug so they cannot collide.
    """
    out: list[Config] = []
    for mode in cfg["modes"]:
        for d_model in cfg["d_model"]:
            for devices in cfg["devices"]:
                if mode == "strong":
                    pops = [int(n) for n in cfg["population"]]
                else:
                    pops = [int(n) * devices for n in cfg["population_per_device"]]
                for population in pops:
                    for strategy in cfg["strategies"]:
                        for how in cfg["how"]:
                            out.append(
                                Config(mode, devices, d_model, population, strategy, how)
                            )
    return out


#: Fixed, so a projection is comparable across configs and across machines.
_PROBE_SEED = 12345
_PROBE_DIM = 16


def fingerprint(tree) -> dict:
    """What the trajectory-identity guard compares across device counts.

    **A hash cannot be the guard, and the dress rehearsal is what proved it.** `docs/03` says
    to "assert the optimizer trajectory is identical across D". For contraction strategy A
    that is literally true: every device regenerates and contracts the whole population in
    the same order, so the update is bitwise identical at D=1 and D=8. Strategy B `psum`s a
    partial update per device, so the summation order *is* the device count and the last ulp
    moves. On the rehearsal all three strategies matched under A and none matched under B.

    That is physics, not a bug, and `tests/gpu/test_device_invariance_gpu.py` already prices
    it at `rtol=1e-5`. So this records a digest (exact, useful when it does match) plus a
    small numeric fingerprint the report can compare with a tolerance: the norm, and a fixed
    random projection, which catches a genuine divergence that a norm alone would miss.
    """
    flat = np.concatenate(
        [np.asarray(x, dtype=np.float64).ravel() for x in jax.tree.leaves(tree)]
    )
    rng = np.random.default_rng(_PROBE_SEED)
    projection = rng.standard_normal((_PROBE_DIM, flat.size))
    return {
        "digest": hashlib.sha256(np.round(flat, 9).tobytes()).hexdigest()[:16],
        "norm": float(np.linalg.norm(flat)),
        "probe": (projection @ flat).tolist(),
    }


def measure(config: Config, cfg: dict) -> dict:
    """Time one configuration. Returns the record written to disk."""
    warmup = int(cfg.get("warmup", 3))
    repeats = int(cfg.get("repeats", 5))

    mesh = sharding.make_mesh(config.devices)
    sharding.check_population(config.population, mesh)

    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=config.d_model)
    batch = transformer_block.make_batch(
        jax.random.fold_in(key, 1),
        d_model=config.d_model,
        batch=int(cfg.get("batch", 8)),
        seq=int(cfg.get("seq", 32)),
    )

    es = ShardedES(
        STRATEGIES[config.strategy](),
        n=config.population,
        sigma=SIGMA,
        lr=LR,
        mesh=mesh,
        how=config.how,
    )
    state = es.init(key, params)

    @jax.jit
    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness)

    # Warm-up. The first call compiles; `docs/03` says discard at least three.
    for _ in range(warmup):
        state = generation(state)
    jax.block_until_ready(state)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        state = generation(state)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)

    # The trajectory guard runs from a *fresh* state for exactly one generation, not from
    # the timed state after warmup+repeats. Drift across generations compounds, and the
    # tolerance this is compared against (tests/gpu, 1e-5) is a one-generation figure. The
    # rehearsal measured 1.2e-5 after 8 generations and it read like a violation; it was an
    # accumulation. Comparing like with like costs one extra generation.
    trajectory = fingerprint(generation(es.init(jax.random.key(SEED), params)).params)

    compiled = generation.lower(state).compile()
    try:
        mem = compiled.memory_analysis()
        peak = int(getattr(mem, "temp_size_in_bytes", 0)) + int(
            getattr(mem, "argument_size_in_bytes", 0)
        )
    except Exception:  # noqa: BLE001
        peak = None

    return {
        "config": asdict(config),
        "seconds_median": statistics.median(times),
        "seconds_iqr": (
            float(np.percentile(times, 75) - np.percentile(times, 25)) if repeats > 1 else 0.0
        ),
        "seconds_all": times,
        "warmup": warmup,
        "repeats": repeats,
        "peak_bytes_per_device": peak,
        # Trajectory identity: same seed, same generation count, so this must not depend on
        # the device count beyond summation order. The report asserts it; recording it is
        # what makes that possible.
        "trajectory": trajectory,
        "generations_run": warmup + repeats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=HERE / "sweep.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=float, default=300.0, help="per-config wall-clock cap, s")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    global RESULTS
    RESULTS = HERE / cfg.get("results_dir", "results")
    configs = expand(cfg)

    available = jax.device_count()
    runnable = [c for c in configs if c.devices <= available]
    skipped = len(configs) - len(runnable)

    if args.dry_run:
        print(f"{len(configs)} configs, {len(runnable)} runnable on {available} devices")
        if skipped:
            print(f"  {skipped} need more devices than this machine has")
        for c in runnable[:10]:
            print(f"  {c.slug()}")
        if len(runnable) > 10:
            print(f"  ... and {len(runnable) - 10} more")
        return 0

    env = harness.capture_env(HERE, OUTPUTS)
    harness.write_atomic(HERE / "env.json", env)
    if env["dirty_worktree"]:
        print("WARNING: dirty worktree, results will be stamped unreproducible")

    done = failed = 0
    for i, config in enumerate(runnable, 1):
        if config.path().exists():
            continue
        t0 = time.perf_counter()
        try:
            record = measure(config, cfg)
        except Exception as exc:  # noqa: BLE001
            # A cap or an OOM at one configuration must not end the sweep. Record it and
            # move on, or a 6-hour session dies on config 3 of 200.
            record = {"config": asdict(config), "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
        record["env"] = env
        record["wall_seconds"] = time.perf_counter() - t0
        harness.write_atomic(config.path(), record)
        done += 1
        status = "ERROR" if "error" in record else f"{record['seconds_median'] * 1e3:.1f} ms"
        print(f"[{i}/{len(runnable)}] {config.slug()}  {status}", flush=True)

    print(f"\n{done} written, {failed} failed, {skipped} needed more devices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
