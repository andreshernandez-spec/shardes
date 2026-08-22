"""T7 on a TPU v5e-8: isolated collectives and the regeneration decomposition.

Two short scripts in one session (review 7, hole 2): allreduce_ladder.py on
all 8 chips, then regen_decompose.py on one. Committed placeholder USERNAME
in kernel-metadata.json; SHA pinned below and printed.
"""
import os
import subprocess
import sys
from pathlib import Path

SHA = "PINNED_AT_PUSH"
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
run([sys.executable, "-c",
     "import jax; v=jax.__version__; d=jax.devices();"
     " print(v, len(d), d[0].platform, d[0].device_kind);"
     " assert tuple(map(int, v.split('.')[:2])) >= (0, 11);"
     " assert len(d) == 8 and d[0].platform == 'tpu'"])
run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])
run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])

env = {**os.environ, "PYTHONPATH": "src"}
r1 = run([sys.executable, "experiments/phase2/allreduce_ladder.py"], env=env, check=False)
r2 = run([sys.executable, "experiments/phase2/regen_decompose.py"], env=env, check=False)

run(["bash", "-c", "cp -r experiments/phase2/results-ladder experiments/phase2/results-regen "
     "/kaggle/working/ && ls /kaggle/working/results-ladder /kaggle/working/results-regen"])
ladder = list(Path("/kaggle/working/results-ladder").glob("*.json"))
regen = list(Path("/kaggle/working/results-regen").glob("*.json"))
print(f"ladder files: {len(ladder)}, regen cells: {len(regen)}")
sys.exit(0 if (r1.returncode == 0 and r2.returncode == 0 and ladder and len(regen) == 4) else 1)
