"""T6: E10's in-context check, minutes of TPU. t1sweep's shape."""

import os
import subprocess
import sys

SHA = "SET-ME"
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


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
r = run([sys.executable, "experiments/phase2/barrier_context.py"], env=env, check=False)
sys.exit(r.returncode)
