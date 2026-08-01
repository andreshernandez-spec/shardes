"""T2': the two-GPU half of Gate G1 criterion 2, run headlessly on Kaggle.

Runs `pytest tests/gpu -m gpu` on 2x T4. Two of those tests cannot be answered anywhere
else: simulated devices share a memory space and never actually communicate, so a
collective that is wrong over a real interconnect still passes on CPU-8.

Expect **16 passed**. Fewer means the checkout is broken, not that the code is fine:
the reference fixture skips rather than fails when it cannot find reference.json.

This is a Kaggle *script* kernel, so there are no `!` magics. Everything is subprocess.
"""

import os
import subprocess
import sys

SHA = "64baf67"  # SET THIS to the commit under test.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"failed with {r.returncode}: {' '.join(cmd)}")
    return r


# Kaggle ships jax 0.7.2 (measured by the probe kernel, 2026-08-01). contraction.py does
# `from jax import shard_map`, which is 0.8+, and pyproject floors it at 0.11. This is the
# only reason the kernel needs internet.
run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]>=0.11"])

run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

# PYTHONPATH rather than an install: tests/ and experiments/phase1/reference.json live
# outside src/shardes, so a wheel install leaves pytest with nothing to collect.
env = {
    **os.environ,
    "PYTHONPATH": "src",
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
}

run(
    [
        sys.executable,
        "-c",
        "import jax; v=jax.__version__;"
        " assert tuple(map(int,v.split('.')[:2]))>=(0,11), f'jax {v} < 0.11';"
        " from jax import shard_map; d=jax.devices();"
        " print(v, len(d), d[0].device_kind);"
        " assert len(d)==2, f'{len(d)} devices, want 2'",
    ],
    env=env,
)

# Collection count first. 16 is the sign the checkout is intact.
run([sys.executable, "-m", "pytest", "tests/gpu", "-m", "gpu", "-q", "--collect-only"], env=env)

result = run(
    [sys.executable, "-m", "pytest", "tests/gpu", "-m", "gpu", "-q", "-s", "-rs"],
    env=env,
    check=False,
)
print(f"\nT2' pytest exit code: {result.returncode}")
sys.exit(result.returncode)
