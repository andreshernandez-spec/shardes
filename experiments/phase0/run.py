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
import json
import os
import platform
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from shardes import metrics
from shardes.strategies.registry import FULL, STRATEGIES, Entry, check_entry

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# The grid the real registry is expected to hold, used ONLY by --dry-run while it is still
# empty, so the pipeline and the figure can be rehearsed before any strategy exists. Note
# the absence of full-rank sobol: that is the non-rectangular grid from docs/01 C0.5, and
# check_entry enforces it here too.
DRY_RUN_GRID = {
    "iid_full": Entry(lambda: None, FULL, "iid"),
    "mirrored_full": Entry(lambda: None, FULL, "mirrored"),
    "mirrored_hd_full": Entry(lambda: None, FULL, "mirrored+orthogonal_hd"),
    "iid_lr1": Entry(lambda: None, 1, "iid"),
    "mirrored_lr1": Entry(lambda: None, 1, "mirrored"),
    "mirrored_hd_lr1": Entry(lambda: None, 1, "mirrored+orthogonal_hd"),
    "mirrored_sobol_lr1": Entry(lambda: None, 1, "mirrored+sobol"),
}


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


def expand(cfg: dict, registry: dict | None = None) -> list[Config]:
    """Cross the config axes with the strategy registry. Deterministic order.

    Sorted by population so a truncated sweep still has complete small-N curves rather
    than a ragged edge across every N.
    """
    axes = cfg["axes"]
    registry = STRATEGIES if registry is None else registry
    out = []
    for name, entry in sorted(registry.items()):
        check_entry(name, entry)
        for population in sorted(axes["population"]):
            for sigma in axes["sigma"]:
                for shaping in axes["shaping"]:
                    out.append(
                        Config(
                            strategy=name,
                            rank=entry.rank,
                            scheme=entry.scheme,
                            population=population,
                            sigma=float(sigma),
                            shaping=shaping,
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


def capture_env() -> dict:
    """Written by the driver, never by hand. Reconstructing it afterwards is never
    accurate, and for a paper it has to be exact (docs/06)."""
    devices = jax.devices()
    dirty = bool(_git("status", "--porcelain"))
    return {
        "commit": _git("rev-parse", "HEAD"),
        # A number from a dirty tree is not reproducible. Record it rather than trust
        # that nobody ever runs a sweep with uncommitted edits, because everyone does.
        "dirty_worktree": dirty,
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        "numpy": np.__version__,
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


def load_estimator():
    """The estimator is the library's, not the driver's (CLAUDE.md ground rules).

    Imported lazily so --dry-run works before it exists.
    """
    from shardes.estimator import estimate  # noqa: PLC0415

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
    base = jax.random.fold_in(jax.random.key(config.seed), hash(config.slug()) % (2**31))

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
    ap.add_argument("--config", type=Path, default=HERE / "config.toml")
    ap.add_argument("--dry-run", action="store_true", help="synthetic numbers")
    ap.add_argument("--limit", type=int, default=None, help="stop after N configs")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.config.read_text())
    registry = STRATEGIES
    if args.dry_run and not STRATEGIES:
        registry = DRY_RUN_GRID
        print("registry is empty; --dry-run is using the expected grid from docs/01 C0.5")
    configs = expand(cfg, registry)

    if args.list:
        for c in configs:
            print(("done " if c.path().exists() else "todo ") + c.slug())
        print(f"\n{len(configs)} configs, {sum(c.path().exists() for c in configs)} done")
        return 0

    if not configs:
        print("no configs: the strategy registry is empty, see "
              "src/shardes/strategies/registry.py", file=sys.stderr)
        return 1

    estimate = synthetic_estimate if args.dry_run else load_estimator()
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
