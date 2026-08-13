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
import json
import hashlib
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import rankdata
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

# One definition of "which populations does this model size get", shared with the memory
# audit. Two copies would drift, and the copy that drifted would be the one that decided
# whether a rented node ran the configuration it was booked for.
from feasible import per_device_bytes, populations  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.iid_gaussian import IIDGaussian  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

#: What this driver writes, and therefore what does not count against the worktree being
#: clean. The config's own `results_dir` is appended in main(): it used to be this hardcoded
#: list alone, so a config naming anything else (`results-calibration`) had its results
#: counted as foreign untracked files and every record stamped itself unreproducible.
OUTPUTS = ("results", "results-rehearsal", "env.json", "figures", "figures-rehearsal")

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
                for population in populations(cfg, mode, d_model, devices):
                    for strategy in cfg["strategies"]:
                        for how in cfg["how"]:
                            out.append(
                                Config(mode, devices, d_model, population, strategy, how)
                            )
    return out


#: XLA:GPU picks reduction algorithms per shape unless told not to. A vmap over N members
#: and a vmap over N/D are different shapes, so the same computation is free to use
#: different arithmetic at different device counts, and the trajectory guard reads that as
#: the library diverging.
DETERMINISM_FLAG = "--xla_gpu_deterministic_ops=true"

#: XLA:GPU builds its command buffers as CUDA graphs, and on more than one device the
#: capture fails outright: "Failed to add memset node to a CUDA graph:
#: CUDA_ERROR_INVALID_VALUE". Setting the flag to the empty value turns command buffers off.
COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="

#: The empty value is the whole point, so a substring test is not enough:
#: `--xla_gpu_enable_command_buffer=FUSION` contains `COMMAND_BUFFER_FLAG` and leaves command
#: buffers on. Only an empty value, at the end of the string or before a space, disables them.
_COMMAND_BUFFER_OFF = re.compile(re.escape(COMMAND_BUFFER_FLAG) + r"(?:\s|$)")


def require_gpu_flags() -> int:
    """Refuse to benchmark on a GPU without `DETERMINISM_FLAG`. Measured, not defensive.

    On 2x T4 at `d=256, N=64`, `lowrank_r1/A`: without the flag `D=1` and `D=2` disagreed by
    6.3e-03 while every repeat within a process was bitwise identical, so it looked like a
    logic bug rather than noise. With the flag every comparison is exactly zero. Worse, two
    processes on the same node disagreed with each other (0.0 and 8.9e-03) for the same
    configuration, which is consistent with autotuning choosing by measured kernel time:
    cached within a process, free to differ between them.

    A guard whose verdict depends on which process it ran in cannot certify a sweep, so this
    is an error rather than a warning. `docs/06`'s G1 cell has always set the flag; the
    phase 2 kernels never did, which is why `tests/gpu` passed while `check.py` failed on
    the same property.

    Setting it here is not possible: XLA reads `XLA_FLAGS` once when the backend
    initialises, and this module imports jax at import time. It has to be in the environment
    before the process starts.

    **`COMMAND_BUFFER_FLAG` is required too, and only when the node has more than one
    device.** Measured on 2x A100-SXM4-80GB (driver 550.127.05, jax 0.11.0): every one of
    the 16 `D=2` rehearsal configs died with "Failed to add memset node to a CUDA graph",
    and every `D=1` config passed. It is not the determinism flag's doing, since `D=2` fails
    with `XLA_FLAGS` empty as well; disabling command buffers fixes it and costs 0.2% of
    step time (13.00 ms -> 13.03 ms at d=512 N=256).

    That failure mode is why this is an error and not a warning. The driver records a failed
    config and keeps going, so a sweep on 8 GPUs would run to completion with its 64 `D=1`
    configs intact and all 192 sharded ones errored, which looks like a finished sweep.

    **It is required on every multi-device GPU, not only where it has been seen to fail.**
    2x T4 passed `tests/gpu` without it on 2026-08-01 (docs/06 G1), so this is not universal
    to CUDA and the requirement is broader than the measurement. Kept broad deliberately:
    there is nothing readable at startup that says whether this node's XLA will capture its
    graphs, the flag costs 0.2% where it is not needed, and the alternative is finding out
    on a rented 8-GPU node.
    """
    platform = jax.devices()[0].platform
    if platform not in ("gpu", "cuda", "rocm"):
        return 0

    flags = os.environ.get("XLA_FLAGS", "")
    missing = []
    if DETERMINISM_FLAG not in flags:
        missing.append(DETERMINISM_FLAG)
    # Single-GPU runs cannot reach the broken path: a config needing more devices than the
    # node has is skipped before it runs. Requiring the flag there would block correct work.
    if len(jax.devices()) > 1 and not _COMMAND_BUFFER_OFF.search(flags):
        missing.append(COMMAND_BUFFER_FLAG)
    if not missing:
        return 0

    print(
        f"refusing to benchmark on {platform} without {' and '.join(missing)}.\n\n"
        f"{DETERMINISM_FLAG} : without it XLA picks reduction algorithms per shape, so D=1\n"
        "and D=2 run different arithmetic and the trajectory guard fails on configurations\n"
        "where the library is correct.\n\n"
        f"{COMMAND_BUFFER_FLAG} : without it every D>1 config dies in CUDA graph capture,\n"
        "and the sweep still exits 0 with its single-device configs intact.\n\n"
        "Re-run with:\n\n"
        f'    XLA_FLAGS="{DETERMINISM_FLAG} {COMMAND_BUFFER_FLAG}" python run.py ...\n\n'
        "The timings this produces are with deterministic reductions and no command buffers,\n"
        "which is the only configuration whose correctness can be checked; say so when\n"
        "quoting them.",
        file=sys.stderr,
    )
    return 2


#: Fixed, so a projection is comparable across configs and across machines.
_PROBE_SEED = 12345
_PROBE_DIM = 16


def fingerprint(tree) -> dict:
    """What the trajectory-identity guard compares across device counts.

    **A hash cannot be the guard, and the dress rehearsal is what proved it.** `docs/03` says
    to "assert the optimizer trajectory is identical across D". Under strategy A the update
    is bitwise identical at D=1 and D=8, but not because the arithmetic is: the fitness
    differs by an ulp or so at every device count, since a vmap over N members and one over
    N/D are different shapes and XLA reduces them differently. `centered_ranks` reads only
    the ordering, so it discards that, and a configuration whose members are close enough for
    an ulp to reorder them breaks A with nothing wrong in the contraction. `rank_digest` is
    what tells those two apart. Strategy B `psum`s a partial update per device, so the
    summation order *is* the device count and the last ulp moves. On the rehearsal all three
    strategies matched under A and none matched under B.

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


def rank_digest(fitness) -> dict:
    """The ordering the shaping actually sees, so a guard failure can name its own cause.

    `fingerprint` records the update. When strategy A's update is *not* bitwise identical
    across device counts, that on its own cannot say which of two things happened, and they
    want opposite responses:

    - the contraction is wrong, which is a bug and the reason the guard exists;
    - the configuration cannot separate its members, so an ulp of arithmetic noise reorders
      them and the rank transform turns that into a different update. That is the documented
      noise floor (`noisefloor.py`), a limitation to write down rather than a bug to fix.

    Recording the ordering splits them. Same ranks with a different update is the first;
    different ranks is the second.

    **Midranks, not `argsort(argsort(f))`**, because midranks are what `centered_ranks` uses.
    Exactly tied members share a rank, so permuting them does not change the update, and
    counting that as a reordering would report a divergence the shaping already absorbed.

    Measured on 8x A100 at `d=512, N=1024, lowrank_r1`: 189 of 1024 fitness entries differ
    between `D=1` and `D=8` by up to 2.00 ulp, 4 members change rank, and the update moves
    4.41e-05. The fitness differing is **not** the anomaly. It differs in the configurations
    that pass too, and rank shaping discards it. Ranks moving is the anomaly.
    """
    f = np.asarray(jax.device_get(fitness), dtype=np.float64).ravel()
    mid = rankdata(f, method="average")
    return {
        "digest": hashlib.sha256(mid.tobytes()).hexdigest()[:16],
        "n": int(f.size),
        # Exact ties are harmless under midranks and informative anyway: a configuration
        # producing them is one where float32 has run out of room to order the population.
        "ties": int((np.diff(np.sort(f)) == 0.0).sum()),
    }


def measure(config: Config, cfg: dict) -> dict:
    """Time one configuration. Returns the record written to disk.

    **Timing and correctness need different matmul precisions, and conflating them cost a
    TPU prep run.** A TPU MXU multiplies in bf16 unless told otherwise, exactly as an Ampere
    GPU uses TF32. Run the trajectory guard at that default and strategy B's updates diverge
    across device counts by up to 3.9e-2, which reads as a sharding bug and is not one: it is
    ~1e-3 relative arithmetic, amplified by a psum whose summation order changes with D.

    So the guard always runs at `highest`, and the timed generations run at whatever
    `matmul_precision` the config asks for, which is recorded in the result. A throughput
    number is only meaningful next to the precision that produced it, and a correctness check
    is only meaningful above the noise floor. One global setting cannot serve both.
    """
    warmup = int(cfg.get("warmup", 3))
    repeats = int(cfg.get("repeats", 5))
    precision = cfg.get("matmul_precision", "highest")

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

    # The guard gets its own jitted function, identical except that it also returns the
    # fitness, because `rank_digest` needs the ordering and `generation` does not expose it.
    # Deliberately not folded into `generation`: that is the function being *timed*, and
    # adding an output to it would mean the timed program is no longer the one the library
    # runs. Benchmark fidelity is worth more here than the compilation.
    #
    # It costs one extra compile per configuration when `matmul_precision` is already
    # "highest", which is what sweep.yaml sets. Measured at ~8-13 s against a ~40 s
    # per-config wall time on 8x A100, so budget roughly +25% on a full re-run.
    @jax.jit
    def guard_generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness), fitness

    with jax.default_matmul_precision(precision):
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
    #
    # `guard_generation` is only ever called here, inside this context, so the guard's
    # precision does not depend on jit's cache key the way it used to. That is why it is a
    # separate function and not just a second call: the previous form relied on entering a
    # different precision context retracing an already-compiled `generation`, which is true
    # but is a subtle thing for a correctness guard to rest on.
    with jax.default_matmul_precision("highest"):
        guard_state, guard_fitness = guard_generation(
            es.init(jax.random.key(SEED), params)
        )
        trajectory = fingerprint(guard_state.params)
        ranks = rank_digest(guard_fitness)

    # **Inside the precision context, and it was not.** `matmul_precision` is part of jit's
    # cache key, so compiling out here produced a DEFAULT-precision program while the timed
    # one ran at whatever the config set, usually "highest". M6 was then reporting the memory
    # of an executable the sweep never ran. Same reason the guard needs its own context.
    with jax.default_matmul_precision(precision):
        compiled = generation.lower(state).compile()
    try:
        mem = compiled.memory_analysis()
        # **Outputs count, minus whatever aliases an input.** This was `temp + argument` and
        # omitted `output - alias`, which for the d=2048 block is about 96 MiB of parameters
        # the updated state has to hold. That undercount lands hardest on exactly the
        # strategies M6 exists to praise: `seed_regenerated` reports 15 MiB per device, so a
        # missing 96 MiB is not a rounding error, it is the result.
        peak = (
            int(getattr(mem, "temp_size_in_bytes", 0))
            + int(getattr(mem, "argument_size_in_bytes", 0))
            + int(getattr(mem, "output_size_in_bytes", 0))
            - int(getattr(mem, "alias_size_in_bytes", 0))
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
        "matmul_precision": precision,
        "guard_precision": "highest",
        # Trajectory identity: same seed, same generation count, so this must not depend on
        # the device count beyond summation order. The report asserts it; recording it is
        # what makes that possible.
        "trajectory": trajectory,
        # The ordering behind that update, so `check.py` can tell a contraction bug from a
        # configuration sitting under the noise floor. See `rank_digest`.
        "ranks": ranks,
        # Which shaping ran, because strategy A's *bitwise* invariance is a property of rank
        # shaping rather than of the contraction: `centered_ranks` reads only the ordering,
        # so it discards the sub-rank fitness differences that device count introduces. Under
        # `centered` or `none` those differences reach the update and A stops being exact.
        # Nobody passes this, so nothing recorded it, so the guard could not know.
        "shaping": getattr(es.shaping, "__name__", repr(es.shaping)),
        "generations_run": warmup + repeats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=HERE / "sweep.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hbm", type=float, default=None,
                    help="per-device HBM in GB; warns about configs that cannot fit")
    ap.add_argument("--cap", type=float, default=300.0,
                    help="per-config wall-clock cap, s: flagged after the fact, see below")
    ap.add_argument("--budget", type=float, default=8 * 3600.0,
                    help="stop scheduling once the session has used this many seconds")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    global RESULTS
    RESULTS = HERE / cfg.get("results_dir", "results")
    configs = expand(cfg)

    rc = require_gpu_flags()
    if rc:
        return rc

    available = jax.device_count()
    runnable = [c for c in configs if c.devices <= available]
    skipped = len(configs) - len(runnable)

    # Memory preflight. Strategy A regenerates all N on every device, so a configuration can
    # be far too large for the node while looking modest in the config file. Reported here
    # rather than discovered as an OOM two hours into a rented session.
    if args.hbm:
        budget = args.hbm * 1024**3 * 0.9
        over = [c for c in runnable
                if per_device_bytes(c.d_model, c.population, c.devices, c.how) > budget]
        if over:
            print(f"WARNING: {len(over)} of {len(runnable)} configs exceed {args.hbm:g} GB "
                  f"per device and will be recorded as errors. Run feasible.py.")
            for c in over[:3]:
                print(f"  {c.slug()}")

    if args.dry_run:
        # **Pending, not runnable.** This reported "256 runnable" for a results directory that
        # already held 256 files, so a dry run before renting hardware looked identical
        # whether the session would measure everything or nothing. Pending is the number that
        # decides whether the session is worth paying for.
        pending, retry = [], 0
        for c in runnable:
            if not c.path().exists():
                pending.append(c)
                continue
            try:
                if "error" in json.loads(c.path().read_text()):
                    pending.append(c)
                    retry += 1
            except (json.JSONDecodeError, OSError):
                pending.append(c)
                retry += 1
        print(f"{len(configs)} configs, {len(runnable)} runnable on {available} devices, "
              f"{len(pending)} PENDING")
        if retry:
            print(f"  {retry} of those are previous errors, which are retried")
        if skipped:
            print(f"  {skipped} need more devices than this machine has")
        if not pending:
            print(f"\nNOTHING TO DO: every configuration already has a result in "
                  f"{cfg.get('results_dir', 'results')}/. Point results_dir at a fresh "
                  "directory, or delete what should be re-measured. Renting hardware for "
                  "this config would measure nothing.")
            return 1
        for c in pending[:10]:
            print(f"  {c.slug()}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return 0

    env = harness.capture_env(HERE, (*OUTPUTS, cfg.get("results_dir", "results")))
    harness.write_atomic(HERE / "env.json", env)
    if env["dirty_worktree"]:
        print("WARNING: dirty worktree, results will be stamped unreproducible")

    # `docs/03` asks for "a hard wall-clock cap per configuration, and exceeding it logs and
    # moves on". Half of that is not implementable in process: a generation is one blocking
    # XLA call, and neither SIGALRM nor a thread can preempt it, so a cap cannot *stop* a
    # configuration that is already running. Pretending otherwise is worse than not having it.
    #
    # What is implementable, and what actually protects a 9-hour session:
    #   - a session budget checked between configurations, so an overrunning sweep stops
    #     scheduling rather than being killed by the session cap with nothing written,
    #   - the cap recorded per configuration, so an overrun is visible in the results,
    #   - resume, so the next session continues instead of starting over.
    # The remaining exposure is one pathological configuration hanging forever. Compile times
    # are measured in the rehearsal, which is what that rehearsal is for.
    started = time.perf_counter()
    done = failed = over = 0
    stopped_early = None
    for i, config in enumerate(runnable, 1):
        # **An error file is not a result and must not be skipped.** A failed configuration
        # writes `{"config": ..., "error": ...}` to the same path a good one uses, so the
        # resume logic used to treat it as done: re-running a sweep that half failed skipped
        # every failure, printed "0 written", and exited 0. Errors are now retried, which is
        # what makes the resume story true rather than nearly true.
        if config.path().exists():
            try:
                if "error" not in json.loads(config.path().read_text()):
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # unreadable: treat as absent and re-measure
        elapsed = time.perf_counter() - started
        if elapsed > args.budget:
            stopped_early = (i, elapsed)
            break
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
        record["cap_seconds"] = args.cap
        record["over_cap"] = record["wall_seconds"] > args.cap
        harness.write_atomic(config.path(), record)
        done += 1
        over += record["over_cap"]
        status = "ERROR" if "error" in record else f"{record['seconds_median'] * 1e3:.1f} ms"
        flag = "  OVER CAP" if record["over_cap"] else ""
        print(f"[{i}/{len(runnable)}] {config.slug()}  {status}{flag}", flush=True)

    print(f"\n{done} written, {failed} failed, {over} over cap, {skipped} needed more devices")
    if stopped_early:
        i, elapsed = stopped_early
        print(f"STOPPED at {i}/{len(runnable)} after {elapsed / 3600:.2f} h "
              f"(budget {args.budget / 3600:.2f} h). Re-run to continue; results resume.")
    # **Non-zero when anything failed or was left undone.** This returned 0 unconditionally,
    # so a sweep that errored on every configuration exited successfully and the only signal
    # was a line of stdout nobody reads on a detached run.
    if failed or stopped_early:
        print("EXITING NON-ZERO: the results directory is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
