#!/usr/bin/env python
"""E1 — estimator quality sweep. Answers Gate G0.

Resumable and idempotent: one JSON file per config, written the moment that config
finishes, and a re-run skips whatever is already on disk. A 20-hour sweep that dies at
hour 19 should cost one config, not everything.

    python run.py --dry-run       # synthetic numbers, exercises the whole pipeline
    python run.py                 # the real sweep
    python run.py --limit 5       # first 5 outstanding configs, for a rehearsal

The estimator itself is NOT here. This driver owns config expansion, resume, timing,
environment capture, results IO and aggregation over replicates. It calls an
`estimate(config, key) -> (g_hat, grad)` supplied by the library. Until that exists,
--dry-run is the only mode that runs.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Must precede `import jax`: XLA reads XLA_FLAGS once, when the backend initializes.
#
# jaxlib 0.11.0 hits a fatal CHECK in XLA's Triton GEMM fusion on this sweep:
#
#   F triton_tiling_propagation.cc:698]
#   Check failed: src_fragment_it != src_fragments_order.end()
#
# It is an abort() inside the compiler, not a Python exception, so `run_one`'s try/except
# cannot contain it: the whole process dies and the sweep stops. Deterministic, and it first
# fires on mirrored_full at N=256, about a quarter of the way in.
#
# Disabling the fusion is measured to be numerically free and not free in time. Re-running
# three completed cells with it off changed cosine_median by 3e-6 to 5e-5 relative, which is
# f32 reassociation noise roughly 100x below the R=30 sampling IQR, and cost 1.14x to 1.51x
# wall clock. Set here rather than in a shell wrapper so a future run cannot forget it and
# crash at hour four; `capture_env` records the result either way.
_TRITON_WORKAROUND = "--xla_gpu_enable_triton_gemm=false"
if _TRITON_WORKAROUND not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_TRITON_WORKAROUND}".strip()

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from shardes import metrics, shaping
from shardes.dimensions import FULL, sampling_dimension
from shardes.strategies.registry import STRATEGIES, check_entry

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# The two sides of the conditional shaping axis (docs/01 C0.5). A scheme is "mirrored" if it
# composes Mirrored, which every coupled scheme does.
SHAPING_SIDES = ("iid", "mirrored")
SHAPING_NAMES = frozenset(shaping.BY_NAME)

# Two axes are conditional, both for reasons in docs/01 C0.5 rather than for tidiness.
POPULATION_SIDES = ("full", "low")


def population_for(rank: int | str, axis: dict) -> list[int]:
    """The populations that apply at `rank`.

    Full rank stops earlier, and it is the figure that says so rather than the clock: the
    full-rank panel exists to show curves not separating at N/d_eff << 1, and it never
    reaches 1 at any N that fits on one GPU. The measured half of the argument is in
    config.yaml and docs/04 C3.3: full-rank orthogonal_hd costs >400 s per replicate at
    N = 2^18, so that cell cannot reach R = 30.
    """
    return axis["full" if rank == FULL else "low"]


def shaping_for(scheme: str, axis: dict) -> list[str]:
    """The shaping modes that apply to `scheme`.

    Conditional, not crossed, and neither list is a subset of the other. `none` is the right
    unshaped baseline under mirroring because a mirrored pair is centred by construction, and
    a dead arm without it. `centered` is the unbiased low-variance baseline on the iid side
    and over-corrects under mirroring, because the pair already cancels `f_bar` so the
    n/(n-1) factor is applied twice. docs/01 C0.5 carries both measurements.
    """
    return axis["mirrored" if "mirrored" in scheme else "iid"]


# ---------------------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    strategy: str
    rank: int | str
    scheme: str
    population: int
    sigma: float
    shaping: str
    replicates: int
    seed: int

    def slug(self) -> str:
        """Stable filename. Sigma is formatted, not repr'd, so 0.001 does not become
        '0.001' on one platform and '1e-03' on another and orphan the earlier result."""
        return (
            f"strategy={self.strategy}"
            f"__N={self.population}"
            f"__sigma={self.sigma:.0e}"
            f"__shaping={self.shaping}"
        )

    def path(self) -> Path:
        return RESULTS / f"{self.slug()}.json"


def load_config(path: Path) -> dict:
    """Load and validate. YAML 1.1 coerces more than it looks, and always silently.

    Two traps this guards, both of which produce a running sweep with wrong values rather
    than an error:

    `no`, `No`, `NO`, `off`, `on`, `yes` are **booleans**, and `~`/`null` are None. A
    shaping mode written bare as `no` arrives as `False`, and every result file is then
    labelled `shaping=False`. Bare `y`/`n` happen to be safe in PyYAML, which omits them
    from the YAML 1.1 bool set, but do not rely on that.

    `1e-3` is a **string**, not a float: YAML 1.1 wants a decimal point (`1.0e-3`). Here
    it happens to survive because sigma is passed through `float()`, but it would not
    survive being compared or formatted, and the failure would surface as a sweep that
    reruns everything because the slugs changed.
    """
    cfg = yaml.safe_load(path.read_text())

    for key in ("seed", "replicates", "wall_clock_cap_s", "axes", "model"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key {key!r}")
    if "kind" not in cfg["model"]:
        raise ValueError(f"{path}: model.kind is required")

    axes = cfg["axes"]
    if not isinstance(axes.get("sigma"), list) or not axes["sigma"]:
        raise ValueError(f"{path}: axes.sigma must be a non-empty list")

    population = axes.get("population")
    if not isinstance(population, dict) or set(population) != set(POPULATION_SIDES):
        raise ValueError(
            f"{path}: axes.population must be a mapping with keys "
            f"{sorted(POPULATION_SIDES)}, got {population!r}. Full rank stops earlier than "
            "low rank: it never reaches N/d_eff = 1, and full-rank orthogonal_hd cannot "
            "reach R=30 at N=2^18 on one GPU. docs/01-phase0-estimator-harness.md C0.5."
        )
    for side, values in population.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: axes.population.{side} must be a non-empty list")

    shaping = axes.get("shaping")
    if not isinstance(shaping, dict) or set(shaping) != set(SHAPING_SIDES):
        raise ValueError(
            f"{path}: axes.shaping must be a mapping with keys {sorted(SHAPING_SIDES)}, got "
            f"{shaping!r}. The shaping axis is conditional on the scheme, not crossed with "
            "it: `centered` over-corrects under mirroring and `none` is a dead arm without "
            "it. docs/01-phase0-estimator-harness.md C0.5."
        )
    for side, values in shaping.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: axes.shaping.{side} must be a non-empty list")
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    f"{path}: axes.shaping.{side} contains {value!r} "
                    f"({type(value).__name__}). PyYAML reads no/No/NO/off/on/yes as booleans "
                    "and ~/null as None; quote the value."
                )
            if value not in SHAPING_NAMES:
                raise ValueError(
                    f"{path}: axes.shaping.{side} contains {value!r}, which is not a shaping "
                    f"mode. Known: {sorted(SHAPING_NAMES)}. Caught here rather than as a "
                    "KeyError partway through a 20-hour sweep."
                )

    chunk = cfg.get("chunk")
    if chunk is not None:
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise ValueError(f"{path}: chunk must be a positive int or absent, got {chunk!r}")
        if chunk % 2:
            raise ValueError(
                f"{path}: chunk must be even, got {chunk}. Mirrored pairs members as "
                "(2k, 2k+1), so an odd chunk splits a pair and loses the antithetic "
                "cancellation. Every mirrored scheme in the registry would raise."
            )

    for side, values in population.items():
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"{path}: axes.population.{side} contains {value!r}, want a positive int"
                )
    for value in axes["sigma"]:
        try:
            sigma = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{path}: axes.sigma contains {value!r}. YAML 1.1 needs a decimal point "
                "in scientific notation: write 1.0e-3, not 1e-3."
            ) from None
        if not sigma > 0:
            raise ValueError(f"{path}: axes.sigma contains {value!r}, want a positive number")

    return cfg


def expand(cfg: dict, registry: dict | None = None) -> list[Config]:
    """Cross the config axes with the strategy registry. Deterministic order.

    Sorted by population so a truncated sweep still has complete small-N curves rather
    than a ragged edge across every N.

    Two axes are selected by the entry rather than crossed with it: `population` by rank and
    `shaping` by scheme. Both are non-rectangular for reasons in docs/01 C0.5.
    """
    axes = cfg["axes"]
    registry = STRATEGIES if registry is None else registry
    out = []
    for name, entry in sorted(registry.items()):
        check_entry(name, entry)
        for population in sorted(population_for(entry.rank, axes["population"])):
            for sigma in axes["sigma"]:
                for mode in shaping_for(entry.scheme, axes["shaping"]):
                    out.append(
                        Config(
                            strategy=name,
                            rank=entry.rank,
                            scheme=entry.scheme,
                            population=population,
                            sigma=float(sigma),
                            shaping=mode,
                            replicates=int(cfg["replicates"]),
                            seed=int(cfg["seed"]),
                        )
                    )
    return out


# ---------------------------------------------------------------------------------------
# Environment, captured not typed
# ---------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def worktree_is_dirty() -> bool:
    """Tracked edits, or untracked files that are not this sweep's own output.

    `git status --porcelain` on its own counts the results directory, and `capture_env` runs
    once at startup, so a *resumed* sweep would see the previous session's untracked results
    and stamp `dirty_worktree: True` on everything it went on to write. The first 70 cells of
    the real E1 run recorded False and the resume recorded True from identical tracked code,
    which is how this was found.

    Results are outputs, not provenance. Everything else still counts, including untracked
    files: a new module that a strategy imports is exactly the kind of thing that makes a
    number unreproducible, and it would not show up in `git diff`.

    Fails safe: if git cannot answer, report dirty. An unknown provenance is not a clean one.
    """
    # `_git` swallows a failure and returns "" for it, which is also what a clean tree looks
    # like, so the repo probe has to be `rev-parse` rather than `status`. No repo means no
    # provenance.
    root = _git("rev-parse", "--show-toplevel")
    if not root or root == "unknown" or not Path(root).is_dir():
        return True
    try:
        skip = f"{Path(RESULTS).relative_to(root)}/"
    except ValueError:
        skip = None  # results live outside the repo; then nothing is exempt

    # --untracked-files=all, because the default collapses a wholly-untracked directory to
    # its shortest prefix: a tree whose only untracked content is results/ reports
    # `?? experiments/`, which no results-prefix filter can match. Treating that prefix as
    # clean would also hide a genuine untracked `experiments/phase0/new_script.py`. Listing
    # every file makes the filter exact, and makes the answer independent of the user's
    # status.showUntrackedFiles setting.
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status == "unknown":
        return True

    # Filtered here rather than with a pathspec: `_git` runs with cwd=HERE, so a `-- .`
    # pathspec would have scoped the check to experiments/phase0 and stopped noticing edits
    # under src/, which is the opposite of the point.
    def counts(line: str) -> bool:
        path = line[3:].strip().strip('"')
        return skip is None or not path.startswith(skip)

    return any(counts(line) for line in status.splitlines())


def capture_env() -> dict:
    """Written by the driver, never by hand. Reconstructing it afterwards is never
    accurate, and for a paper it has to be exact (docs/06)."""
    devices = jax.devices()
    dirty = worktree_is_dirty()
    return {
        "commit": _git("rev-parse", "HEAD"),
        # A number from a dirty tree is not reproducible. Record it rather than trust
        # that nobody ever runs a sweep with uncommitted edits, because everyone does.
        "dirty_worktree": dirty,
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        "numpy": np.__version__,
        # Sobol direction numbers are read out of scipy rather than vendored, so the scipy
        # version is part of what a sobol result depends on.
        "scipy": getattr(__import__("scipy"), "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "device_count": len(devices),
        "device_kind": getattr(devices[0], "device_kind", "unknown"),
        "device_platform": devices[0].platform,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
    }


# ---------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------


def params_spec(cfg: dict):
    """Shape-and-dtype tree for the configured model, without building it.

    d_eff is computed from this rather than from live params, so --dry-run records the
    same number a real run would and the rehearsal figure has the right x-axis.
    """
    m = cfg["model"]
    kind = m["kind"]
    if kind == "quadratic":
        return jax.ShapeDtypeStruct((int(m["d"]),), jnp.float32)
    if kind == "transformer_block":
        d = int(m["m"])
        if int(m["n"]) != d:
            raise ValueError("the Phase 0 block is square; model.m must equal model.n")
        return {
            name: jax.ShapeDtypeStruct((d, d), jnp.float32)
            for name in ("wq", "wk", "wv", "wo", "w_up", "w_down")
        }
    raise NotImplementedError(f"no params spec for model.kind={kind!r}")


def build_problem(cfg: dict, key):
    """(model, params, batch, true_gradient) for the configured objective.

    `model(params, batch) -> scalar` and the gradient is exact: analytic for the
    quadratic, backprop for the block. No proxy metric and no reference estimator with a
    huge N, which is the point of using a differentiable model (docs/01 C0.4).

    The MLP is not written. It sits between the two in realism and adds nothing the block
    does not already cover for G0.
    """
    from shardes.problems import quadratic, transformer_block  # noqa: PLC0415

    model_cfg = cfg["model"]
    kind = model_cfg["kind"]

    if kind == "quadratic":
        d = int(model_cfg["d"])
        q = quadratic.make(key, d, condition_number=float(model_cfg.get("condition_number", 10.0)))
        theta = jax.random.normal(jax.random.fold_in(key, 1), (d,), dtype=jnp.float32)
        return (lambda p, _b: quadratic.value(q, p)), theta, None, quadratic.grad(q, theta)

    if kind == "transformer_block":
        d = int(model_cfg["m"])
        params = transformer_block.init(key, d_model=d)
        batch = transformer_block.make_batch(
            jax.random.fold_in(key, 1),
            d_model=d,
            batch=int(model_cfg.get("batch", 8)),
            seq=int(model_cfg.get("seq", 32)),
        )
        return (
            transformer_block.loss,
            params,
            batch,
            transformer_block.grad(params, batch),
        )

    raise NotImplementedError(
        f"model.kind={kind!r} is not implemented. 'quadratic' and 'transformer_block' are. "
        "src/shardes/problems/mlp.py is still docstring-only."
    )


def make_estimator(cfg: dict):
    """Adapt the library estimator to the driver's Config.

    The library takes (strategy, model, params, x, key, ...) and must not import this
    driver's dataclass, so the adaptation lives here rather than there.

    **Jitted, and it matters more than it looks.** Unjitted, every replicate re-traces the
    whole estimator: two `lax.scan`s over chunks plus the strategy's few hundred primitives.
    Measured on the 3080 at N = 1024, chunk = 256, the tracing floor was ~2 s against ~0.6 s
    of actual work. Across 504 configs x 30 replicates that is hours of pure overhead.

    `sigma` is a traced argument rather than static, so the sigma axis costs no extra
    compilations: 12 strategies x 7 populations x 2 shapings, not x 3 again. `params` and
    `batch` are arguments rather than closure constants, so a 6 MB params tree is not
    embedded in every executable.
    """
    from shardes.estimator import estimate as library_estimate  # noqa: PLC0415

    key = jax.random.key(int(cfg["seed"]))
    model, params, batch, truth = build_problem(cfg, key)
    chunk = cfg.get("chunk")

    @functools.partial(jax.jit, static_argnames=("strategy", "population", "shaping_name"))
    def compiled(params, batch, replicate_key, sigma, *, strategy, population, shaping_name):
        return library_estimate(
            STRATEGIES[strategy].build(), model, params, batch, replicate_key,
            member_ids=jnp.arange(population),
            sigma=sigma,
            shaping=shaping.BY_NAME[shaping_name],
            chunk=chunk,
        )

    def estimate(config: Config, replicate_key):
        g_hat = compiled(
            params, batch, replicate_key, jnp.float32(config.sigma),
            strategy=config.strategy,
            population=config.population,
            shaping_name=config.shaping,
        )
        return g_hat, truth

    return estimate


def synthetic_estimate(config: Config, key):
    """Plausibly-shaped fake numbers for the dress rehearsal.

    Shaped so the figure looks like a real one: accuracy improves with N, and the coupled
    schemes separate from iid only at low rank. That is deliberately the *positive* G0
    outcome, because the point of the rehearsal is to check the plot can render the case
    we would most want to read carefully. It is not evidence of anything.
    """
    d = 64
    grad = jax.random.normal(jax.random.key(config.seed), (d,))
    n_eff = config.population / 1024.0
    err = 1.0 / np.sqrt(1.0 + n_eff)
    if config.rank != "full" and "orthogonal_hd" in config.scheme:
        err *= 0.6
    if config.rank != "full" and "sobol" in config.scheme:
        err *= 0.7
    noise = jax.random.normal(key, (d,)) * err * float(jnp.linalg.norm(grad)) / np.sqrt(d)
    return grad + noise, grad


def run_one(config: Config, estimate, cap_s: float) -> dict:
    """R replicates, aggregated. Median and IQR, never a single number."""
    started = time.time()

    # Common random numbers: every config uses the same replicate seeds, so a difference
    # between two curves is the method rather than the draw. This is a paired comparison
    # and it is worth a lot when R is only 30.
    #
    # The previous version folded in `hash(config.slug())`. Two bugs in one line: it
    # broke the pairing, and CPython salts str hashes per process, so a resumed sweep
    # would silently have used different seeds from the run it was resuming.
    base = jax.random.key(config.seed)

    cosines, mses = [], []
    g_sum = None
    grad = None
    truncated = False

    for r in range(config.replicates):
        g_hat, grad = estimate(config, jax.random.fold_in(base, r))
        cosines.append(float(metrics.cosine_similarity(g_hat, grad)))
        mses.append(float(metrics.relative_mse(g_hat, grad)))
        g_sum = g_hat if g_sum is None else jax.tree.map(jnp.add, g_sum, g_hat)

        # Checked *after* a replicate, so a config always yields at least one sample.
        # Checking first means a tight cap produces a config with no data at all, which
        # is worse than overrunning. Interrupting mid-XLA-call would need a subprocess
        # or a signal handler and neither is reliable enough to be worth it; overrunning
        # by one replicate is not the failure this guard is for.
        if time.time() - started > cap_s:
            truncated = True
            break

    if not cosines:
        raise RuntimeError(f"{config.slug()}: no replicates completed within {cap_s}s")

    done = len(cosines)
    mean_g = jax.tree.map(lambda x: x / done, g_sum)

    return {
        "config": asdict(config),
        "replicates_completed": done,
        "truncated": truncated,
        "cosine_median": float(np.median(cosines)),
        "cosine_q1": float(np.percentile(cosines, 25)),
        "cosine_q3": float(np.percentile(cosines, 75)),
        "relative_mse_median": float(np.median(mses)),
        "relative_mse_q1": float(np.percentile(mses, 25)),
        "relative_mse_q3": float(np.percentile(mses, 75)),
        "relative_bias": float(metrics.relative_bias(mean_g, grad)),
        "wall_clock_s": time.time() - started,
    }


def write_atomic(path: Path, payload: dict) -> None:
    """Write to a temp file and rename.

    A partial JSON left by a kill mid-write would still *exist*, so resume would skip it
    forever and the sweep would silently carry a corrupt config. Rename is atomic on the
    same filesystem, so a file either is not there or is complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="synthetic numbers")
    ap.add_argument("--limit", type=int, default=None, help="stop after N configs")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # One registry for both modes. There used to be a separate rehearsal grid, because the
    # registry held fewer strategies than F5 needs panels for and a --dry-run against it
    # would have left the low-rank panel unexercised. The registry now covers every cell of
    # docs/01 C0.5, asserted by test_registry.py, so the two would be copies of each other.
    configs = expand(cfg, STRATEGIES)

    if args.list:
        for c in configs:
            print(("done " if c.path().exists() else "todo ") + c.slug())
        print(f"\n{len(configs)} configs, {sum(c.path().exists() for c in configs)} done")
        return 0

    if not configs:
        print("no configs: the strategy registry is empty, see "
              "src/shardes/strategies/registry.py", file=sys.stderr)
        return 1

    estimate = synthetic_estimate if args.dry_run else make_estimator(cfg)
    spec = params_spec(cfg)
    env = capture_env()
    env["dry_run"] = args.dry_run
    if env["dirty_worktree"]:
        print("WARNING: uncommitted changes; these results are not reproducible",
              file=sys.stderr)

    outstanding = [c for c in configs if not c.path().exists()]
    # Count before --limit truncates, or configs deferred by the limit get reported as
    # finished and a resumed sweep looks further along than it is.
    already_done = len(configs) - len(outstanding)
    if args.limit:
        outstanding = outstanding[: args.limit]
    print(f"{len(configs)} configs, {already_done} already done, "
          f"running {len(outstanding)}")

    started = time.time()
    failures = []
    for i, config in enumerate(outstanding, 1):
        t = time.time()
        try:
            record = run_one(config, estimate, float(cfg["wall_clock_cap_s"]))
        except Exception as exc:  # noqa: BLE001
            # One bad config must not cost the session. No result file is written, so a
            # later run retries it: an OOM at N = 2^18 may well succeed on a quieter box,
            # and a config silently marked done would be worse than one retried.
            failures.append((config.slug(), repr(exc)))
            print(f"[{i}/{len(outstanding)}] FAILED {config.slug()}: {exc}",
                  file=sys.stderr)
            continue

        record["env"] = env
        # Recorded, not reconstructed. plot.py used to rebuild d_eff from a CLI flag,
        # which silently went wrong the moment the model had more than one matrix.
        record["d_eff"] = sampling_dimension(spec, config.rank)
        # Loud, so a synthetic file can never be mistaken for a measurement.
        record["SYNTHETIC"] = args.dry_run
        write_atomic(config.path(), record)
        print(f"[{i}/{len(outstanding)}] {config.slug()}  "
              f"cos={record['cosine_median']:.4f}  {time.time() - t:.1f}s")

    env["total_wall_clock_s"] = time.time() - started
    env["failures"] = failures
    write_atomic(HERE / "env.json", env)

    if failures:
        print(f"\n{len(failures)} config(s) failed and were not written; re-run to retry",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
