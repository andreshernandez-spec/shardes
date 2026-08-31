"""E17b on a TPU v5e-8: the real-model crossover grid for the figure.

Same shape as e17tpu (which ran E17's 31 cells in one session): upgrade jax,
assert 8 chips in a fresh interpreter, clone at the pinned SHA, run the two
short phase2 jobs (the regeneration decomposition and the contraction
isolation, both bounded), then the driver, up to four times under a 1.5 h
budget each (session 2 was killed mid-cell and a relaunch resumes over the
per-cell files), and bring results-regen, results-contraction and
results-e17b home. Exit code: 0 when the driver finished the grid, 2 on a
budget stop, 1 otherwise; either way, resume by committing the results and
pushing again at a SHA that contains them.
"""
import subprocess
import sys
from pathlib import Path

SHA = "PINNED_AT_PUSH"
# Per invocation, not per session: the retry loop below can run the driver
# several times. Session 4 spent four slices to add 25 cells and stopped with
# the loop exhausted, not the grid, so six slices now fill the 9 h session.
# 10 cells remain, 3 of them immediate OOMs (>= 30 members per device).
BUDGET_S = "5400"
ATTEMPTS = 6


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
run([sys.executable, "-m", "pip", "install", "-q",
     "transformers", "huggingface_hub", "safetensors", "pyyaml"])
probe = subprocess.run(
    [sys.executable, "-c",
     "import jax; d=jax.devices(); print(len(d), d[0].platform)"],
    capture_output=True, text=True, check=True)
n, platform = probe.stdout.split()
print(f"devices: {n} platform: {platform}", flush=True)
assert platform == "tpu" and int(n) == 8, f"want 8 tpu, got {n} {platform}"

run(["git", "clone", "https://github.com/andreshernandez-spec/shardes.git"])
run(["git", "checkout", "-q", SHA], cwd="shardes")
run(["git", "log", "--oneline", "-1"], cwd="shardes")
run([sys.executable, "-m", "pip", "install", "-q", "-e", "shardes", "--no-deps"])

# Small phase2 jobs ride this queue slot rather than waiting hours for their
# own; the TPU allows one session at a time, so a separate kernel for a
# ten-minute job would cost this grid its place in the queue. Session 1
# (41b04b9) ran the collective ladder, committed since. Two run here:
#
#   regen_decompose        the v5e re-measurement under the timer fixed in
#                          6718d82, which the sliced-timer records superseded
#   contraction_isolation  C per cell, the open term in docs/11's cost model
#
# Both are budgeted and resumable per cell, and neither blocks the grid: a
# failure or a budget stop prints and moves on. 25 min total ceiling against
# a 9 h session, and the low-rank cells (the ones the crossover needs) run
# first, so a short stop still lands the useful half.
reg = subprocess.run([sys.executable, "shardes/experiments/phase2/regen_decompose.py"])
print(f"regen exit: {reg.returncode}", flush=True)
con = subprocess.run([sys.executable,
                      "shardes/experiments/phase2/contraction_isolation.py",
                      "--budget", "1200"])
print(f"contraction exit: {con.returncode} (2 = budget stop, resumable)", flush=True)
subprocess.run(["bash", "-c", "cp -r shardes/experiments/phase2/results-regen "
                "shardes/experiments/phase2/results-contraction . || true"])

# Session 2 was killed mid-cell with no message after a run of recorded OOMs
# (host memory, not HBM: the driver catches RESOURCE_EXHAUSTED and records it).
# Every finished cell is on disk, so relaunching resumes over the skip list.
# Stop when the grid is done, when the driver asks to stop (budget, code 2),
# or when an entire invocation adds nothing, which is the real failure.
CELLS = Path("shardes/experiments/countdown/results-e17b")
prev = -1
for attempt in range(ATTEMPTS):
    r = subprocess.run(
        [sys.executable, "shardes/experiments/countdown/e17_systems.py",
         "--config", "shardes/experiments/countdown/e17b.yaml", "--budget", BUDGET_S])
    have = len(list(CELLS.glob("*.json")))
    print(f"attempt {attempt}: driver exit {r.returncode}, cells on disk {have}",
          flush=True)
    # Exit 2 is this invocation's slice ending, not the session's: the budget is
    # per invocation so the loop can bound host memory, which is what killed
    # session 2. Session 3 broke here and spent 1.5 h of a 9 h slot. Stop only
    # when the grid is done or an invocation adds nothing.
    if have == 128 or have == prev:
        break
    prev = have

run(["bash", "-c", "cp -r shardes/experiments/countdown/results-e17b . && ls results-e17b"])
count = len(list(Path("results-e17b").glob("*.json")))
print(f"cells written: {count} of 128", flush=True)
sys.exit(r.returncode if r.returncode else (0 if count == 128 else 1))
