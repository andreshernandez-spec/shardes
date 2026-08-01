"""How many GPUs does the Kaggle API actually give us?

The UI offers "GPU T4 x2". The API documents only singular accelerator IDs
(NvidiaTeslaT4, NvidiaTeslaP100, ...) and says nothing about count. T2' is only
interesting on two devices, so this has to be settled before the real run: a
one-GPU session would produce "9 skipped" and look like a pass.

Needs no internet and no install. Counts devices the two ways that can disagree,
because JAX_PLATFORMS and CUDA_VISIBLE_DEVICES can hide a device that nvidia-smi
still lists.
"""

import os
import subprocess

print("== nvidia-smi ==")
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "(no output)")

print("== env ==")
for k in ("CUDA_VISIBLE_DEVICES", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE"):
    print(f"{k}={os.environ.get(k, '(unset)')}")

print("== torch ==")
try:
    import torch

    print(f"torch {torch.__version__}, cuda.device_count()={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
except Exception as exc:  # noqa: BLE001
    print(f"torch unavailable: {exc}")

print("== jax (preinstalled version, whatever it is) ==")
try:
    import jax

    print(f"jax {jax.__version__}, devices={jax.devices()}")
except Exception as exc:  # noqa: BLE001
    print(f"jax unavailable: {exc}")

print("== VERDICT ==")
n = len(
    [
        line
        for line in subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.splitlines()
        if line.strip().startswith("GPU")
    ]
)
print(f"GPU_COUNT={n}")
print("T2' is runnable through the API" if n >= 2 else "T2' needs the browser UI, not the API")
