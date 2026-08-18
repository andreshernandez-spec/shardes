"""T3: the E10 shaping-barrier measurement on a TPU v5e-8.

t1sweep's shape: upgrade jax, assert 8 chips in a fresh interpreter, clone at a
pinned SHA, run the driver under an internal budget. The grid is small (84
cells of microsecond-scale ops; compile time dominates), so the budget is slack
rather than an expected stop.
"""

import os
import subprocess
import sys

SHA = "SET-ME"  # the commit under test; must contain barrier.py and barrier-tpu.yaml.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"
BUDGET_S = "7200"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
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
r = run([sys.executable, "experiments/phase2/barrier.py",
         "--config", "experiments/phase2/barrier-tpu.yaml", "--budget", BUDGET_S],
        env=env, check=False)

sys.exit(r.returncode)
