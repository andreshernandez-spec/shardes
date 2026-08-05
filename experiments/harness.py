"""Provenance shared by every experiment driver: what ran, from which commit, on what.

Extracted from `phase0/run.py` when phase 2 needed the same thing. It is deliberately not a
copy: the two functions here have specific bug history, and a second copy would drift from
the first exactly where it matters least visibly.

- `worktree_is_dirty` once flagged every *resumed* sweep as dirty, because a sweep's own
  untracked results counted against it. The first 70 cells of E1 recorded False and the
  resume recorded True from identical tracked code.
- The same function later missed edits under `src/`, because a `-- .` pathspec scoped the
  check to the driver's own directory.

Both fixes live here now. `docs/conventions.md` records that every gap mutation testing has
found so far was a duplicated fact; this is the cheap way not to add another.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import jax
import numpy as np


def git(here: Path, *args: str) -> str:
    """`git` in `here`, or the string "unknown". Never raises.

    **`rstrip`, not `strip`.** `git status --porcelain` puts the index status in column 1 and
    the worktree status in column 2, so a file modified or deleted but not staged starts with
    a *space*: `" D path"`. Stripping both ends ate that space on the first line only, and
    `worktree_is_dirty` then read the path from offset 3 and lost a character, so it matched
    no output prefix and every run reported dirty.

    It hid for a long time because the case it was written for is untracked files, which
    start `"?? "` with nothing to strip. It surfaced when a sweep deleted results to
    regenerate them: 228 deletions, the first one misparsed, and every regenerated result
    stamped unreproducible.
    """
    try:
        return subprocess.run(
            ["git", *args], cwd=here, capture_output=True, text=True, timeout=10
        ).stdout.rstrip()
    except Exception:
        return "unknown"


def worktree_is_dirty(
    here: Path, outputs: Iterable[str], git_fn: Callable[..., str] | None = None
) -> bool:
    """Tracked edits, or untracked files that are not this driver's own output.

    `outputs` names what the driver writes (results, env.json, figures). Outputs are not
    provenance: a rerun that overwrites its own results has not changed the code that
    produced them, and counting them makes every resumed sweep flag itself unreproducible.
    Everything else counts, including untracked files: a new module a strategy imports is
    exactly what makes a number unreproducible and would not show up in `git diff`.

    Fails safe. If git cannot answer, report dirty; an unknown provenance is not a clean one.

    `git_fn` is injectable so a driver's tests can simulate git failing.
    """
    g = git_fn or (lambda *a: git(here, *a))

    # `git` returns "" for a failure, which is also what a clean tree looks like, so the
    # repo probe has to be `rev-parse` rather than `status`. No repo means no provenance.
    root = g("rev-parse", "--show-toplevel")
    if not root or root == "unknown" or not Path(root).is_dir():
        return True
    try:
        base = Path(here).relative_to(root)
        skip = tuple(str(base / o) for o in outputs)
    except ValueError:
        skip = None  # outputs live outside the repo; then nothing is exempt

    # --untracked-files=all, because the default collapses a wholly-untracked directory to
    # its shortest prefix: a tree whose only untracked content is results/ reports
    # `?? experiments/`, which no results-prefix filter can match.
    status = g("status", "--porcelain", "--untracked-files=all")
    if status == "unknown":
        return True

    # Filtered here rather than with a pathspec: git runs with cwd=here, so a `-- .`
    # pathspec would scope the check to the driver's directory and stop noticing edits
    # under src/, which is the opposite of the point.
    #
    # Whole path components, not a string prefix. `startswith` exempted anything merely
    # *starting* with an output's name, so `results-calibration/` was silently covered by
    # `results` and `env.json.bak` by `env.json`. That is the wrong direction to be wrong in:
    # a stray file is exactly the kind of thing that makes a number unreproducible, and it
    # was being hidden by a name collision.
    def counts(line: str) -> bool:
        path = line[3:].strip().strip('"')
        if skip is None:
            return True
        return not any(path == o or path.startswith(o + "/") for o in skip)

    return any(counts(line) for line in status.splitlines())


def capture_env(
    here: Path, outputs: Iterable[str], git_fn: Callable[..., str] | None = None
) -> dict:
    """Written by the driver, never by hand. Reconstructing it afterwards is never accurate,
    and for a paper it has to be exact (docs/06)."""
    g = git_fn or (lambda *a: git(here, *a))
    devices = jax.devices()
    return {
        "commit": g("rev-parse", "HEAD"),
        # A number from a dirty tree is not reproducible. Record it rather than trust that
        # nobody runs a sweep with uncommitted edits, because everyone does.
        "dirty_worktree": worktree_is_dirty(here, outputs, git_fn),
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


def write_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file in the same directory, then rename.

    A sweep that is killed mid-write must not leave a half-written JSON that a resume then
    reads as a completed configuration. Rename is atomic on the same filesystem, so a file either is not there or is complete.
    `indent=2, sort_keys=True` matches what phase 0 already wrote, so extracting this did not
    reformat a single committed result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
