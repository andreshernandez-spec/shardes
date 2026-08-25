"""The v5e half of the cost model: contraction isolation and the regen re-measure.

Two short phase2 jobs on a TPU v5e-8, in one session, in cheapest-first order:

  contraction_isolation.py   C, C/D, the in-situ all-reduce and shard_ratio at the
                             F2 grid's cells. Measured on 8xA100 2026-08-25; the v5e
                             half is what the paper's section 6 says is missing.
  regen_decompose.py         the four v5e cells superseded by the sliced-timer bug
                             (results-regen/sliced-timer/, fixed in 6718d82).

WHY A DEDICATED KERNEL. These rode the e17btpu kernel so they would not cost E17b its
place in the one-session-at-a-time queue. That kernel is in ERROR with 45 of 128 grid
cells left, so riding it means the paper's missing half waits on a campaign that needs
several more sessions. This runs the two jobs alone in about half an hour and leaves
E17b to resume separately.

BUDGETS. Both jobs are bounded and resume per cell, so a session that dies partway costs
one cell. contraction_isolation gets 2400 s: on the A100 the full 20-cell grid took 12
minutes, and the v5e is slower per cell but hits its 16 GB ceiling earlier, and an OOM
cell is recorded and skipped rather than retried. regen_decompose gets what is left.

THE 16 GB CEILING IS DATA. iid_gaussian at d=2048 materializes ~25.8 GB of noise for
strategy A's replicated contraction; that fits an 80 GB A100 and cannot fit a v5e chip.
The driver records the failure as the cell's outcome. Cells that OOM are a result about
where the memory ceiling sits, not a failed session, so the exit code below counts
written files rather than successes.

Exit: 0 if both jobs wrote every cell they could, 2 on a budget stop (incomplete is not
complete), 1 otherwise. Resume by committing what came back and pushing at a SHA that
contains it.
"""
import os
import subprocess
import sys
from pathlib import Path

SHA = "PINNED_AT_PUSH"
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"
CONTRACTION_BUDGET = "2400"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
# Fresh interpreter: the process that ran the install already imported the old jax and
# initialized a backend, so an in-process check proves nothing.
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
r1 = run([sys.executable, "experiments/phase2/contraction_isolation.py",
          "--budget", CONTRACTION_BUDGET], env=env, check=False)
print(f"contraction exit: {r1.returncode} (2 = budget stop, resumable)", flush=True)
r2 = run([sys.executable, "experiments/phase2/regen_decompose.py"], env=env, check=False)
print(f"regen exit: {r2.returncode}", flush=True)

run(["bash", "-c", "cp -r experiments/phase2/results-contraction "
     "experiments/phase2/results-regen /kaggle/working/ && "
     "ls /kaggle/working/results-contraction /kaggle/working/results-regen"])
cells = list(Path("/kaggle/working/results-contraction").glob("*.json"))
regen = list(Path("/kaggle/working/results-regen").glob("*.json"))
print(f"contraction cells: {len(cells)} of 20, regen cells: {len(regen)} of 4", flush=True)

if r1.returncode == 2:
    sys.exit(2)
sys.exit(0 if len(cells) == 20 and len(regen) == 4 else 1)
