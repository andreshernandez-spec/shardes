"""Whatever the sweeps benchmark, the real-hardware invariance suite must also check.

Split out of `test_accelerator_coverage.py`, which is otherwise about `tests/gpu` alone and
stays with the library. This one reads the sweep configs, so it moves with the experiments
when the repository splits (docs/14). After that it needs another way to learn which
strategies the library guards on real hardware, because `tests/gpu` will be in a different
repository; docs/14 lists that as open.
"""

import importlib.util
import pathlib
import sys

GPU_TEST = (pathlib.Path(__file__).resolve().parent / "gpu"
            / "test_device_invariance_gpu.py")


def _load():
    spec = importlib.util.spec_from_file_location("gpu_invariance", GPU_TEST)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_invariance"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_benchmarked_strategy_is_also_guarded():
    """The sweep and the invariance test must cover the same strategies.

    They did not. `lowrank_r1` was in `sweep.yaml` and in neither `tests/gpu` nor
    `rehearsal.yaml`, so it was benchmarked on rented hardware by a suite that never
    checked it was device-count invariant, and the dress rehearsal that certified the
    driver never ran it. It is also the only strategy the sweep has failed on.

    Subset rather than equality: guarding a strategy the sweep does not run is fine, the
    reverse is not.
    """
    import yaml

    phase2 = GPU_TEST.parent.parent.parent / "experiments" / "phase2"
    guarded = set(_load().NAMES)
    for name in ("sweep.yaml", "rehearsal.yaml"):
        wanted = set(yaml.safe_load((phase2 / name).read_text())["strategies"])
        assert wanted <= guarded, (
            f"{name} benchmarks {sorted(wanted - guarded)}, which tests/gpu never checks "
            "for device-count invariance"
        )
