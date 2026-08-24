"""E17b on a TPU v5e-8: the real-model crossover grid for the figure.

Same shape as e17tpu (which ran E17's 31 cells in one session): upgrade jax,
assert 8 chips in a fresh interpreter, clone at the pinned SHA, run the
driver under a 7 h internal budget, bring results-e17b home. Exit code: 0
when the driver finished the grid, 2 on a budget stop (resume by committing
the results and pushing again at a SHA that contains them), 1 otherwise.
"""
import subprocess
import sys
from pathlib import Path

SHA = "PINNED_AT_PUSH"
BUDGET_S = "25200"


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

r = subprocess.run(
    [sys.executable, "shardes/experiments/countdown/e17_systems.py",
     "--config", "shardes/experiments/countdown/e17b.yaml", "--budget", BUDGET_S])

run(["bash", "-c", "cp -r shardes/experiments/countdown/results-e17b . && ls results-e17b"])
count = len(list(Path("results-e17b").glob("*.json")))
print(f"cells written: {count} of 128", flush=True)
sys.exit(r.returncode if r.returncode else (0 if count == 128 else 1))
