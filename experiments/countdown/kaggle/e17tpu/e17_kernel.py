"""E17 on a TPU v5e-8: the contraction crossover on the real model.

Committed placeholder USERNAME in kernel-metadata.json; fill at push time.
SHA is pinned below and printed, so the log carries what ran.
"""
import json
import subprocess
import sys

SHA = "PINNED_AT_PUSH"

def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])
run([sys.executable, "-m", "pip", "install", "-q",
     "transformers", "huggingface_hub", "safetensors", "pyyaml"])

# Assert the platform in a FRESH interpreter: this process imported nothing
# yet, but the upgrade must be validated the same way regardless.
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
     "--config", "shardes/experiments/countdown/e17.yaml"])

# Bring the results home: kernels output downloads everything in cwd.
run(["bash", "-c",
     "cp -r shardes/experiments/countdown/results-e17 . && ls results-e17"])

# Fail loudly if the driver did: status ERROR is the one visible signal.
count = len(list(__import__("pathlib").Path("results-e17").glob("*.json")))
print(f"cells written: {count}", flush=True)
sys.exit(r.returncode if r.returncode else (0 if count == 32 else 1))
