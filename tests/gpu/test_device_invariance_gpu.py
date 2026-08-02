"""Real-hardware check that the simulated-device shortcut did not lie. Gate G1 criterion 2.

    XLA_FLAGS=--xla_gpu_deterministic_ops=true pytest tests/gpu -m gpu -q -s

Marked `gpu`, deselected by default (`pyproject.toml` addopts), run by hand before each gate.
docs/02: do this **before** Phase 2, not during. On one GPU the portability half runs and the
sharded half skips; two or more GPUs runs everything.

---

**Two different claims, two different tolerances, and conflating them is the trap.**

`SHARDING_RTOL` — real GPUs against each other across `D`, and `A` against `B`. This is
invariant 2 and it is a *correctness* claim: the same seed must produce the same update
however the population is split. Tight, because only summation order changes.

`PLATFORM_RTOL` — GPU against the committed CPU-8 reference. This is a *portability* claim
and can only ever be loose: CPU and GPU run different kernels and reduce in different orders.
One tolerance covering both would either let a real sharding bug through or fail on
arithmetic that was never promised to match.

TF32 is the specific thing that would make this look broken. Left at its default an Ampere
GPU computes matmuls at roughly 1e-3 relative accuracy, which reads exactly like a
device-invariance failure and is not one. `jax_default_matmul_precision="highest"` below is
load-bearing rather than defensive tuning.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

PHASE1 = pathlib.Path(__file__).resolve().parent.parent.parent / "experiments" / "phase1"
REFERENCE = PHASE1 / "reference.json"

#: Same platform, different device counts, or A vs B. Summation order only.
SHARDING_RTOL = 1e-5
#: GPU against the CPU-simulated reference. Different kernels, different reduction trees.
PLATFORM_RTOL = 1e-4


#: Any real accelerator, not just a GPU. **A TPU has to be in here or this whole file
#: silently becomes a no-op on one**: every test gates on this list, so an omitted platform
#: means 16 skips, exit 0, and a green run that checked nothing. That is the same failure as
#: an accelerator request being silently downgraded, and harder to notice.
_REAL = ("gpu", "cuda", "rocm", "tpu")


def _accelerators():
    return [d for d in jax.devices() if d.platform in _REAL]


#: Kept as the old name so nothing that imports it breaks. The marker is still `gpu` because
#: it is wired into pyproject's addopts; read it as "needs real hardware".
_gpus = _accelerators


@pytest.fixture(scope="module", autouse=True)
def highest_precision():
    """Without this an Ampere GPU uses TF32 and every comparison below fails at ~1e-3.

    It matters at least as much on a TPU, where the MXU multiplies in bf16 by default. Same
    symptom, different hardware: a portability failure that is really a precision default.
    """
    old = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_default_matmul_precision", old)


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE} missing; run experiments/phase1/reference.py on CPU")
    return json.loads(REFERENCE.read_text())


def _run(strategy_name: str, n_devices: int, how: str) -> np.ndarray:
    """The same generation the reference recorded, on real devices.

    Imported from the generator rather than reimplemented: the seed, population, sigma and
    model are part of the artifact, and a second copy of them here would drift and the drift
    would look like a hardware discrepancy.
    """
    if str(PHASE1) not in sys.path:
        sys.path.insert(0, str(PHASE1))
    import reference as ref  # noqa: PLC0415

    return ref.one_generation(strategy_name, n_devices, how)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


NAMES = ["iid_gaussian", "seed_regenerated", "mirrored_lr1"]


@pytest.mark.skipif(not _accelerators(), reason="no real accelerator visible")
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("how", ["A", "B"])
def test_one_gpu_matches_the_cpu_reference(reference, name, how):
    """The portability half. One real GPU against eight simulated CPU devices.

    Runnable on any single-GPU box, so it does not wait on multi-GPU access, and it is the
    half that catches a *numerical* difference between the shortcut and real hardware as
    opposed to a sharding one.
    """
    want = np.array(reference["updates"][f"{name}/{how}"])
    got = _run(name, 1, how)
    assert _rel(got, want) < PLATFORM_RTOL, f"{name}/{how} differs from the CPU reference"


@pytest.mark.skipif(len(_accelerators()) < 2, reason="needs 2 real GPUs (T2', docs/06)")
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("how", ["A", "B"])
def test_two_gpus_match_one_gpu(name, how):
    """The sharding half, and the one G1 actually asks for. Invariant 2 on real hardware.

    Against *one GPU* rather than against the CPU reference, deliberately: that isolates the
    thing under test. A failure here means the population split changed the answer on
    hardware the simulated devices modelled wrongly — precisely the failure
    `--xla_force_host_platform_device_count` cannot expose, because simulated devices share
    one memory space and never actually communicate.
    """
    one, two = _run(name, 1, how), _run(name, 2, how)
    assert _rel(two, one) < SHARDING_RTOL, f"{name}/{how} differs between 1 and 2 real GPUs"


@pytest.mark.skipif(len(_accelerators()) < 2, reason="needs 2 real GPUs")
@pytest.mark.parametrize("name", NAMES)
def test_the_two_contraction_strategies_agree_on_real_hardware(name):
    """A and B on 2 real GPUs. On CPU they agree partly because nothing is really
    communicated; here the psum and the all-gather are collectives over an interconnect."""
    a, b = _run(name, 2, "A"), _run(name, 2, "B")
    assert _rel(a, b) < SHARDING_RTOL, f"{name}: A and B differ on real hardware"


@pytest.mark.skipif(not _accelerators(), reason="no real accelerator visible")
def test_report_the_environment(reference, capsys):
    """Not an assertion. G1 needs the run recorded, and a green tick naming no hardware is
    not evidence. Run with `-s`."""
    devices = jax.devices()
    with capsys.disabled():
        print(f"\n  reference: {reference['env']}")
        print(f"  hardware:  jax {jax.__version__}, {len(devices)} x "
              f"{getattr(devices[0], 'device_kind', '?')} ({devices[0].platform})")
        print(f"  matmul precision: {jax.config.jax_default_matmul_precision}")
