"""What does Kaggle's TPU tier actually hand you?

Phase 2 needs 8 devices. docs/06 calls Kaggle TPU v5e-8 the primary scaling platform (free,
8 chips); docs/03 assumes 8 rented GPUs. That is $0 against ~$150, so it is worth 90 seconds
of quota to check rather than to assume.

Asking for an accelerator does not mean getting it: on 2026-08-01 a request for
NvidiaTeslaA100 came back a P100, silently. So this reports what is actually present.
"""

import os
import subprocess
import sys


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return f"({cmd[0]} not present)"


print("== env ==")
for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "TPU_NAME", "COLAB_TPU_ADDR"):
    print(f"{k}={os.environ.get(k, '(unset)')}")
print("nvidia-smi:", sh("nvidia-smi", "-L") or "(none)")

print("\n== preinstalled jax ==")
try:
    import jax

    print(f"jax {jax.__version__}")
    print(f"device_count={jax.device_count()}  local={jax.local_device_count()}")
    for d in jax.devices():
        print(f"  {d.platform} {d.device_kind} id={d.id}")
except Exception as exc:  # noqa: BLE001
    print(f"jax failed: {type(exc).__name__}: {exc}")

print("\n== VERDICT ==")
try:
    import jax

    n, kind = jax.device_count(), jax.devices()[0].device_kind
    print(f"DEVICE_COUNT={n}  KIND={kind}  PLATFORM={jax.devices()[0].platform}")
    print("Phase 2 can run free on this tier" if n >= 8 else f"only {n} devices, not 8")
except Exception as exc:  # noqa: BLE001
    print(f"no verdict: {exc}")
