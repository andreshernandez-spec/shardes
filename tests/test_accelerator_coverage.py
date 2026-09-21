"""`tests/gpu` must not become a no-op on hardware we actually run on.

Every test in `tests/gpu` gates on a platform allow-list. A platform missing from that list
does not fail: the tests **skip**, pytest exits 0, and the run reports green having checked
nothing. That is how a TPU sweep could have been certified by a suite that never executed.

These tests run in the *default* suite, on CPU, with no accelerator needed. That is the point.
A guard that only runs on the hardware it is guarding cannot catch the hardware being missing.
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


def test_the_allow_list_covers_every_platform_this_project_targets():
    """The runbook (shardes-paper:docs/06-benchmark-runbook.md) routes work to CUDA GPUs (T2, T2') and to Kaggle TPU v5e-8 (T1).

    If a tier is in the runbook but not here, the gate for that tier passes vacuously.
    """
    covered = set(_load()._REAL)
    for platform in ("gpu", "cuda", "tpu"):
        assert platform in covered, (
            f"{platform!r} is missing from the accelerator allow-list, so every test in "
            "tests/gpu would skip on it and the run would report green having checked nothing"
        )


def test_every_test_in_the_gpu_suite_gates_on_the_shared_predicate():
    """A test that invents its own device check is one nobody will remember to widen."""
    source = GPU_TEST.read_text()
    assert 'd.platform in ("gpu", "cuda", "rocm")' not in source, (
        "an inlined platform tuple has come back; it is the thing that made this suite a "
        "no-op on TPU"
    )
    assert source.count("_accelerators()") >= 4, "tests should gate on the shared predicate"


def test_the_reference_artifact_covers_every_guarded_strategy():
    """`test_one_gpu_matches_the_cpu_reference` reads `reference.json` by key. A name in
    NAMES with no entry there fails at lookup rather than reporting a missing reference,
    so regenerating the artifact is part of adding a strategy."""
    import json

    reference = GPU_TEST.parent.parent.parent / "validation" / "reference.json"
    have = set(json.loads(reference.read_text())["updates"])
    for name in _load().NAMES:
        for how in ("A", "B"):
            assert f"{name}/{how}" in have, (
                f"reference.json has no {name}/{how}; re-run validation/reference.py "
                "on CPU after adding a strategy"
            )


def test_the_suite_is_not_silently_empty():
    """21 is the number the runbook (shardes-paper:docs/06-benchmark-runbook.md) tells a human
    to check. Keep them agreeing."""
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(GPU_TEST), "-m", "gpu", "-q", "--collect-only"],
        capture_output=True, text=True, cwd=GPU_TEST.parent.parent.parent,
        env={"JAX_PLATFORMS": "cpu", "PATH": __import__("os").environ["PATH"],
             "HOME": __import__("os").environ.get("HOME", "")},
    )
    assert "21 tests collected" in out.stdout, (
        f"expected 21 collected; the runbook in shardes-paper quotes that number to the operator.\n{out.stdout[-800:]}"
    )


def test_the_reference_artifact_still_describes_the_code():
    """`validation/reference.json` is what real hardware is held to, so it has to be what
    this code computes on simulated devices. It went stale once and nobody knew: a commit
    that vmapped LowRank's column draws kept the noise bit-identical and still moved three
    rank-1 updates by one float32 ulp, and the only thing that checks, `reference.py
    --check`, is a command a person has to remember to run.

    Not exact equality, which `--check` asks for and which is a statement about one CPU's
    code generation. A reference that is wrong because the sampler, the seeds or the
    contraction changed is wrong by orders of magnitude more than this allows; one that
    differs because a compiler reassociated a sum is not wrong at all.
    """
    import json

    import numpy as np

    validation = GPU_TEST.parent.parent.parent / "validation"
    if str(validation) not in sys.path:
        sys.path.insert(0, str(validation))
    import reference as ref  # noqa: PLC0415

    committed = json.loads((validation / "reference.json").read_text())
    rebuilt = ref.build(committed["config"]["n_devices"])
    assert set(rebuilt["updates"]) == set(committed["updates"])
    for key, update in rebuilt["updates"].items():
        np.testing.assert_allclose(
            np.array(update), np.array(committed["updates"][key]), rtol=1e-5, atol=1e-7,
            err_msg=f"{key}: reference.json no longer matches what the code computes; "
                    "find out why, then re-run validation/reference.py")
