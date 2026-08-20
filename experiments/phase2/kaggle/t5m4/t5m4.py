"""T5: M4 baselines on a TPU v5e, the open half of E9 (TB1's TPU column).

Same shapes as the A100 M4 runs (m4.py --config sweep-tpu.yaml derives the sweep's
four strong cells), D=1 as the primary like-for-like comparison and D=8 as the
what-sharding-adds column, exactly the GPU protocol.

The references: evosax from PyPI; EGGROLL from the authors' repo at the same pin the
GPU run used (b77f7d6), installed --no-deps per docs/03 (their __init__ pulls a torch
model zoo; the noiser under test needs jax and optax only, and m4.py's loader routes
past the __init__). GPL-3.0 stays in its own checkout, never vendored. m4.py records
any unavailable arm in the output rather than silently comparing against less.
"""

import os
import subprocess
import sys

SHA = "SET-ME"  # shardes commit under test
EGGROLL_SHA = "b77f7d6"  # docs/03 M4's pin; the loader is tested against this layout
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
run([sys.executable, "-m", "pip", "install", "-q", "evosax", "optax"])
run(["git", "clone", "-q", "https://github.com/ESHyperscale/HyperscaleES.git",
     "/kaggle/working/HyperscaleES"])
run(["git", "-C", "/kaggle/working/HyperscaleES", "checkout", "-q", EGGROLL_SHA])
run([sys.executable, "-m", "pip", "install", "-q", "-e",
     "/kaggle/working/HyperscaleES", "--no-deps"])
run([
    sys.executable, "-c",
    "import jax; v=jax.__version__; d=jax.devices();"
    " print(v, len(d), d[0].platform, d[0].device_kind);"
    " assert tuple(map(int, v.split('.')[:2])) >= (0, 11), f'jax {v} < 0.11';"
    " assert len(d) == 8, f'{len(d)} devices after the upgrade, want 8';"
    " assert d[0].platform == 'tpu', f'platform is {d[0].platform}, not tpu'",
])

run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

env = {**os.environ, "PYTHONPATH": "src"}
worst = 0
for devices in ("1", "8"):
    r = run([sys.executable, "experiments/phase2/m4.py",
             "--config", "experiments/phase2/sweep-tpu.yaml",
             "--devices", devices,
             "--out", "experiments/phase2/results-m4-tpu-v5e8"],
            env=env, check=False)
    worst = max(worst, r.returncode)

sys.exit(worst)
