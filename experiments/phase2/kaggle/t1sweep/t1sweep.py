"""T1: the Phase 2 sweep on a TPU v5e-8, one session's worth, resumable.

The probe (tpuprep) proved the stack on 2026-08-18: jax upgrades cleanly, the
library runs on 8 real chips, the driver completes rehearsal cells. This kernel
runs the real grid (sweep-tpu.yaml) under an 8 h internal budget: the driver
writes one JSON per cell and stops cleanly at the budget, so a session that ends
incomplete exits non-zero ON PURPOSE and the next session resumes from whatever
the previous one committed. Read the log tail to tell a budget stop from a
crash; the STOPPED line names it.
"""

import os
import subprocess
import sys

SHA = "57b3c81"  # SET THIS to the commit under test; must contain the prior sessions' results.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"
BUDGET_S = "28800"


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
r = run([sys.executable, "experiments/phase2/run.py",
         "--config", "experiments/phase2/sweep-tpu.yaml", "--budget", BUDGET_S],
        env=env, check=False)

# The results are the output either way: kernels output pulls the checkout, and
# the JSONs under results-tpu-v5e8/ are what the next session resumes from once
# committed. Propagate the driver's verdict so status alone says complete or not.
sys.exit(r.returncode)
