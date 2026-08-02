"""The M1/M2/M3 driver's infrastructure: expansion, slugs, resume, and the trajectory guard.

None of this computes a scaling number, which is why it is worth testing. `docs/03` budgets
one rented afternoon and says to treat it as executing an already-debugged plan; a resume bug
does not announce itself, it just makes the sweep "seem to finish early".

The guard tests are the ones that matter most. `check.py` holds contraction strategy A to
bitwise equality across device counts and strategy B to a tolerance, and getting that
backwards fails in opposite, equally silent ways: B held to exact equality never passes, A
held to a tolerance stops noticing a real sharding bug.

Loaded by path, like the phase 0 driver, because experiments/ is not a package.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

PHASE2 = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase2"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"phase2_{name}", PHASE2 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"phase2_{name}"] = mod  # @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


run = _load("run")
check = _load("check")


CFG = {
    "modes": ["strong", "weak"],
    "devices": [1, 2, 4],
    "d_model": [32],
    "population": [16],
    "population_per_device": [4],
    "strategies": ["iid_gaussian"],
    "how": ["A", "B"],
}


# --------------------------------------------------------------------------- expansion


def test_expansion_is_deterministic():
    assert run.expand(CFG) == run.expand(CFG)


def test_strong_holds_the_total_population_and_weak_scales_it():
    configs = run.expand(CFG)
    strong = {c.devices: c.population for c in configs if c.mode == "strong"}
    weak = {c.devices: c.population for c in configs if c.mode == "weak"}
    assert set(strong.values()) == {16}, "strong scaling must fix the total"
    assert weak == {1: 4, 2: 8, 4: 16}, "weak scaling must fix the population per device"


def test_slugs_are_unique():
    configs = run.expand(CFG)
    assert len({c.slug() for c in configs}) == len(configs)


def test_the_mode_is_in_the_slug():
    """Strong and weak collide at the device count where their populations agree."""
    configs = run.expand(CFG)
    collide = [c for c in configs if c.population == 16 and c.devices == 4]
    assert len(collide) == 4, "expected both modes at D=4, N=16"
    assert len({c.slug() for c in collide}) == 4


# --------------------------------------------------------------------------- resume


def test_resume_skips_only_what_is_already_written(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    configs = run.expand(CFG)
    run.harness.write_atomic(configs[0].path(), {"done": True})
    remaining = [c for c in configs if not c.path().exists()]
    assert len(remaining) == len(configs) - 1
    assert configs[0] not in remaining


def test_a_partial_write_is_never_left_behind(tmp_path):
    target = tmp_path / "x.json"
    with pytest.raises(TypeError):
        run.harness.write_atomic(target, {"bad": {1, 2}})  # a set is not JSON
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == [], "a temp file survived the failure"


# --------------------------------------------------------------------------- the guard


def _fingerprint(norm: float, probe: list[float], digest: str = "abc") -> dict:
    return {"digest": digest, "norm": norm, "probe": probe}


def _write(results: pathlib.Path, devices: int, how: str, trajectory: dict, strategy="s"):
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"mode": "strong", "devices": devices, "d_model": 32,
                   "population": 16, "strategy": strategy, "how": how},
        "seconds_median": 1.0,
        "trajectory": trajectory,
    }
    (results / f"D={devices}__how={how}__s={strategy}.json").write_text(json.dumps(payload))


def test_strategy_a_must_be_bitwise_identical_across_devices(tmp_path, capsys):
    """A regenerates and contracts in the same order everywhere, so anything else is a bug."""
    _write(tmp_path, 1, "A", _fingerprint(1.0, [1.0, 2.0], digest="same"))
    _write(tmp_path, 2, "A", _fingerprint(1.0, [1.0, 2.0], digest="different"))
    assert check.main(["--results", str(tmp_path)]) == 1
    assert "not bitwise identical" in capsys.readouterr().out


def test_strategy_b_may_differ_within_tolerance(tmp_path):
    """B psums a partial update, so summation order is the device count. Last-ulp drift is
    expected and must not fail the gate."""
    _write(tmp_path, 1, "B", _fingerprint(1.0, [1.0, 2.0], digest="one"))
    _write(tmp_path, 2, "B", _fingerprint(1.0, [1.0 + 1e-7, 2.0], digest="two"))
    assert check.main(["--results", str(tmp_path)]) == 0


def test_strategy_b_still_fails_on_a_real_divergence(tmp_path, capsys):
    """The tolerance is for summation order, not for a different computation."""
    _write(tmp_path, 1, "B", _fingerprint(1.0, [1.0, 2.0], digest="one"))
    _write(tmp_path, 2, "B", _fingerprint(1.0, [1.5, 2.0], digest="two"))
    assert check.main(["--results", str(tmp_path)]) == 1
    assert "exceeds rtol" in capsys.readouterr().out


def test_the_tolerance_is_defended_at_its_actual_value(tmp_path):
    """A divergence just above 1e-5 must fail.

    Without this, the only thing holding RTOL at 1e-5 is the literal-value check below, and
    a mutation loosening it to 1e-1 survived every behavioural test: the real-divergence case
    above uses a 0.5 relative error, which is caught at any sane tolerance. This one sits in
    the band that only the correct value rejects.
    """
    _write(tmp_path, 1, "B", _fingerprint(1.0, [1.0], digest="one"))
    _write(tmp_path, 2, "B", _fingerprint(1.0, [1.0 + 5e-5], digest="two"))
    assert check.main(["--results", str(tmp_path)]) == 1


def test_the_tolerance_matches_the_gpu_test():
    """Same claim, so the same number. If tests/gpu moves, this has to move with it."""
    gpu_test = (pathlib.Path(__file__).resolve().parent / "gpu"
                / "test_device_invariance_gpu.py").read_text()
    assert "SHARDING_RTOL = 1e-5" in gpu_test
    assert check.RTOL == 1e-5


def test_weak_scaling_rows_are_not_held_to_the_identity_claim(tmp_path):
    """Weak scaling changes N with D on purpose, so its trajectories are meant to differ."""
    for devices in (1, 2):
        results = tmp_path
        results.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {"mode": "weak", "devices": devices, "d_model": 32,
                       "population": 4 * devices, "strategy": "s", "how": "A"},
            "seconds_median": 1.0,
            "trajectory": _fingerprint(float(devices), [float(devices)], digest=str(devices)),
        }
        (results / f"weak-{devices}.json").write_text(json.dumps(payload))
    assert check.main(["--results", str(tmp_path)]) == 0


def test_a_recorded_error_does_not_crash_the_guard(tmp_path, capsys):
    """One OOM must not end a rented session, so errors are recorded and reported, not raised."""
    _write(tmp_path, 1, "A", _fingerprint(1.0, [1.0], digest="x"))
    (tmp_path / "err.json").write_text(
        json.dumps({"config": {"mode": "strong", "devices": 2, "d_model": 32,
                               "population": 16, "strategy": "s", "how": "A"},
                    "error": "RuntimeError: out of memory"})
    )
    assert check.main(["--results", str(tmp_path)]) == 0
    assert "recorded an error" in capsys.readouterr().out


# --------------------------------------------------------------------------- fingerprint


def test_the_probe_catches_a_change_the_norm_would_miss():
    """A norm alone is invariant to sign flips and permutations, which are exactly what a
    sharding bug produces. The fixed random projection is why the guard is not fooled."""
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([-1.0, 2.0, 3.0])  # same norm, different vector
    assert np.isclose(np.linalg.norm(a), np.linalg.norm(b))
    rng = np.random.default_rng(run._PROBE_SEED)
    projection = rng.standard_normal((run._PROBE_DIM, 3))
    assert not np.allclose(projection @ a, projection @ b)


def test_the_projection_is_reproducible_across_calls():
    """A seeded projection, or two runs of the same config would never compare equal."""
    tree = {"w": np.arange(6.0).reshape(2, 3)}
    assert run.fingerprint(tree)["probe"] == run.fingerprint(tree)["probe"]
