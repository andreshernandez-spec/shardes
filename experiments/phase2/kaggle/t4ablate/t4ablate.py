"""T4: E11's session, two drivers in sequence on a TPU v5e.

First the lr1 re-measure: the 28 lr1 cells whose pre-pad records were deleted rerun
under the padded program (cost-sweep-tpu.yaml resumes by file existence, so only
they are pending). Then the precision ablation (cost-precision-tpu.yaml), the one
number E11 still lacks: `highest` vs `default` on identical cells.

Exit code is the worst of the two, so status COMPLETE means both finished.
"""

import os
import subprocess
import sys

SHA = "SET-ME"  # must contain cost-precision-tpu.yaml and the lr1 record deletions.
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
    " assert d[0].platform == 'tpu', f'platform is {d[0].platform}, not tpu'",
])

run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

env = {**os.environ, "PYTHONPATH": "src"}
worst = 0
for config, budget in (("cost-sweep-tpu.yaml", "7200"), ("cost-precision-tpu.yaml", "10800")):
    r = run([sys.executable, "experiments/phase2/cost.py",
             "--config", f"experiments/phase2/{config}",
             "--budget", budget, "--allow-partial"],
            env=env, check=False)
    worst = max(worst, r.returncode)

sys.exit(worst)
