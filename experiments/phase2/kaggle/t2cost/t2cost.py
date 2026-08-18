"""T2: the E8 cost surface on a TPU v5e (one chip), resumable.

Same shape as t1sweep: upgrade jax, assert the stack in a fresh interpreter, clone
at a pinned SHA, run the driver under an internal budget. cost.py is D=1 by design
(the arithmetic-intensity story is per-device), so this uses one of the eight chips
and the assert only cares that the platform is tpu.

--allow-partial because undersized cells are the point on a 16 GB chip: the GPU run
recorded 59 of them on 80 GB and this surface exists to show where the ceiling moves.
A budget stop still exits non-zero (unvisited is not the same as undersized), so
kernel status ERROR means either a stop to resume from or a crash; the log tail's
STOPPED line says which.
"""

import os
import subprocess
import sys

SHA = "SET-ME"  # the commit under test; must contain cost-sweep-tpu.yaml and prior results.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"
BUDGET_S = "21600"


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
    " assert d[0].platform == 'tpu', f'platform is {d[0].platform}, not tpu'",
])

run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

env = {**os.environ, "PYTHONPATH": "src"}
r = run([sys.executable, "experiments/phase2/cost.py",
         "--config", "experiments/phase2/cost-sweep-tpu.yaml",
         "--budget", BUDGET_S, "--allow-partial"],
        env=env, check=False)

# kernels output pulls the whole checkout, so the JSONs come back either way; the
# exit code is what makes status COMPLETE mean "surface finished" from outside.
sys.exit(r.returncode)
