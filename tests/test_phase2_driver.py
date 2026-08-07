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
comms = _load("comms")
profile = _load("profile")


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
    """A's update is bitwise identical across device counts, so anything else has to be
    explained. Not because the arithmetic matches: the fitness differs by an ulp at every
    device count, and `centered_ranks` reads only the ordering and discards it. Which is why
    the guard reaches for the rank digest before calling this a contraction bug."""
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


# --------------------------------------------------------------------------- memory


feasible = _load("feasible")


def test_strategy_a_memory_does_not_fall_with_device_count():
    """The asymmetry that nearly wasted a rented booking.

    A regenerates the whole population on every device, so its storage is `N*|params|` at any
    `D`. B holds one shard. Reading A as if it sharded is what put six 96 GB configurations
    into the first sweep config, including the `D=1` baseline that every parallel-efficiency
    figure divides by.
    """
    # The claim is about what the STRATEGY stores. Total per-device memory also carries
    # activations, which are sharded for both A and B, so the totals are not equal; asserting
    # on them was asserting the old single-term model rather than the property.
    a1 = feasible.perturbation_bytes("iid_gaussian", 2048, 1024)          # A contracts all N
    a8 = feasible.perturbation_bytes("iid_gaussian", 2048, 1024)
    b8 = feasible.perturbation_bytes("iid_gaussian", 2048, 1024 // 8)
    assert a1 == a8, "A must not appear to shard; it regenerates everything everywhere"
    assert b8 * 8 == a8, "B must fall as 1/D"

    # And the total for A must still be dominated by the un-sharded part, or the sizing
    # advice this file exists to give is wrong.
    total_a1 = feasible.per_device_bytes(2048, 1024, 1, "A", "iid_gaussian")
    total_a8 = feasible.per_device_bytes(2048, 1024, 8, "A", "iid_gaussian")
    assert total_a8 > 0.9 * total_a1, "A's memory must barely improve with more devices"


def test_the_strategies_do_not_all_store_the_population():
    """Measured, and the reason the first model was wrong in both directions.

    `iid_gaussian` materialises. `seed_regenerated` re-derives, so its footprint is flat in
    N. The low-rank path stores `N*r*(m+n)` and never an `(m, n)` perturbation, which is
    invariant 3. Predicting one number for all three over-sized two strategies and under-sized
    the one that OOMs.
    """
    d, n = 2048, 512
    iid = feasible.perturbation_bytes("iid_gaussian", d, n)
    seed = feasible.perturbation_bytes("seed_regenerated", d, n)
    low = feasible.perturbation_bytes("lowrank_r1", d, n)
    assert seed == feasible.perturbation_bytes("seed_regenerated", d, 8 * n), "flat in N"
    assert low < iid / 100, "the low-rank path must not scale like a materialised one"
    assert seed < iid / 100


def test_the_committed_sweep_fits_an_80gb_device():
    """`sweep.yaml` is the config the node gets booked for. If it does not fit, the booking
    buys error records."""
    import pathlib as _p

    cfg = __import__("yaml").safe_load(
        (_p.Path(__file__).resolve().parent.parent / "experiments" / "phase2"
         / "sweep.yaml").read_text())
    over = [r for r in feasible.audit(cfg, 80.0) if r[1]]
    assert not over, f"{len(over)} configs exceed 80 GB: {over[:2]}"


def test_populations_may_be_per_model_size_or_shared():
    """A mapping keyed by model size, because the memory ceiling is per model size. A plain
    list still works, which is what the rehearsal config uses."""
    shared = {"population": [16], "population_per_device": [4]}
    per_size = {"population": {512: [16], 2048: [8]}, "population_per_device": {512: [4], 2048: [2]}}
    assert feasible.populations(shared, "strong", 512, 4) == [16]
    assert feasible.populations(per_size, "strong", 2048, 4) == [8]
    assert feasible.populations(per_size, "weak", 2048, 4) == [8]  # 2 per device x 4


def test_a_near_zero_probe_component_does_not_manufacture_a_failure(tmp_path):
    """The metric bug, encoded.

    The guard used elementwise `max(|a-b| / max(|b|, 1e-30))`. A random projection of a
    parameter vector has components near zero routinely, and dividing by one turns a tiny
    absolute difference into an enormous ratio. It reported `lowrank_r1/B` diverging by
    2.7e-02 where the update actually agreed to 2.3e-07, and it failed a TPU run whose cause
    was then misattributed to matmul precision.

    Here the vectors agree to ~1e-9 in norm and differ by 100% on the near-zero component.
    That must pass: the claim is about the update as a vector, not about a coordinate whose
    magnitude is arithmetic noise.
    """
    _write(tmp_path, 1, "B", _fingerprint(1.0, [1.0, 1e-12], digest="one"))
    _write(tmp_path, 2, "B", _fingerprint(1.0, [1.0 + 1e-9, 2e-12], digest="two"))
    assert check.main(["--results", str(tmp_path)]) == 0


def test_the_guard_uses_the_same_measure_as_the_gpu_test():
    """Same claim, same metric. tests/gpu compares norms; so must this."""
    gpu = (pathlib.Path(__file__).resolve().parent / "gpu"
           / "test_device_invariance_gpu.py").read_text()
    assert "np.linalg.norm(a - b) / np.linalg.norm(b)" in gpu
    src = (PHASE2 / "check.py").read_text()
    assert "np.linalg.norm(a - b) / denom" in src
    assert "1e-30" not in src, "the elementwise divide-by-near-zero metric is back"


class _FakeDevice:
    def __init__(self, platform):
        self.platform = platform


def _fake_devices(monkeypatch, platform, count=1):
    monkeypatch.setattr(run.jax, "devices", lambda: [_FakeDevice(platform)] * count)


def test_a_gpu_sweep_without_the_determinism_flag_is_refused(monkeypatch):
    """The flag is load-bearing, not hygiene.

    Measured on 2x T4: without it `lowrank_r1/A` at d=256 N=64 disagreed between D=1 and
    D=2 by 6.3e-03 while repeating bitwise within a process, and two processes on the same
    node gave different answers for the same configuration. With it every comparison is
    exactly zero. A sweep that runs without it produces a guard verdict that depends on
    which process it ran in.
    """
    _fake_devices(monkeypatch, "gpu")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    assert run.require_gpu_flags() != 0


def test_the_determinism_flag_satisfies_the_check_alongside_others(monkeypatch):
    """XLA_FLAGS is a space-separated list, so the check must not require it to stand alone."""
    _fake_devices(monkeypatch, "gpu")
    monkeypatch.setenv(
        "XLA_FLAGS", f"--xla_force_host_platform_device_count=8 {run.DETERMINISM_FLAG}"
    )
    assert run.require_gpu_flags() == 0


def test_cpu_and_tpu_are_not_asked_for_a_gpu_flag(monkeypatch):
    """The flag is CUDA-specific. Requiring it off GPU would block the CPU rehearsal and
    the TPU tier for no reason, and a guard that blocks correct work gets deleted."""
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    for platform in ("cpu", "tpu"):
        _fake_devices(monkeypatch, platform, count=8)
        assert run.require_gpu_flags() == 0


def test_a_multi_gpu_sweep_without_the_command_buffer_flag_is_refused(monkeypatch):
    """The failure this prevents is silent at the sweep level, which is why it is an error.

    Measured on 2x A100-SXM4-80GB: all 16 `D=2` rehearsal configs died in CUDA graph capture
    and all `D=1` configs passed. The driver records a failed config and continues, so an
    8-GPU sweep would exit 0 with 64 results and 192 errors and still look finished.
    """
    _fake_devices(monkeypatch, "gpu", count=2)
    monkeypatch.setenv("XLA_FLAGS", run.DETERMINISM_FLAG)
    assert run.require_gpu_flags() != 0


def test_both_gpu_flags_together_satisfy_the_check(monkeypatch):
    _fake_devices(monkeypatch, "gpu", count=8)
    monkeypatch.setenv(
        "XLA_FLAGS", f"{run.DETERMINISM_FLAG} {run.COMMAND_BUFFER_FLAG}"
    )
    assert run.require_gpu_flags() == 0


def test_a_single_gpu_is_not_asked_for_the_command_buffer_flag(monkeypatch):
    """Only D>1 reaches the broken path, and a config needing more devices than the node has
    is skipped before it runs. A guard that blocks correct single-GPU work gets deleted."""
    _fake_devices(monkeypatch, "gpu", count=1)
    monkeypatch.setenv("XLA_FLAGS", run.DETERMINISM_FLAG)
    assert run.require_gpu_flags() == 0


def test_a_non_empty_command_buffer_value_does_not_satisfy_the_check(monkeypatch):
    """`--xla_gpu_enable_command_buffer=FUSION` contains the flag string and leaves command
    buffers on. The empty value is the whole point, so a substring test is not enough."""
    _fake_devices(monkeypatch, "gpu", count=2)
    monkeypatch.setenv(
        "XLA_FLAGS", f"{run.DETERMINISM_FLAG} {run.COMMAND_BUFFER_FLAG}FUSION"
    )
    assert run.require_gpu_flags() != 0


def test_the_command_buffer_flag_is_recognised_before_another_flag(monkeypatch):
    """It is disabling, so it is normally last in the string. It must still count when it is
    not: the empty value is followed by a space rather than the end of XLA_FLAGS."""
    _fake_devices(monkeypatch, "gpu", count=2)
    monkeypatch.setenv(
        "XLA_FLAGS", f"{run.COMMAND_BUFFER_FLAG} {run.DETERMINISM_FLAG}"
    )
    assert run.require_gpu_flags() == 0


def _env(**over):
    base = {"device_platform": "gpu", "device_kind": "Tesla T4", "jax": "0.11.0",
            "jaxlib": "0.11.0", "commit": "abc1234", "xla_flags": run.DETERMINISM_FLAG}
    return {**base, **over}


def test_the_guard_refuses_results_that_mix_xla_flags(tmp_path, capsys):
    """Same reason as mixing backends: different flags are different arithmetic, and the
    flagged/unflagged pair is the one that actually happened."""
    traj = {"digest": "a", "norm": 1.0, "probe": [1.0, 2.0]}
    for devices, flags in ((1, run.DETERMINISM_FLAG), (2, "")):
        row = {
            "config": {"mode": "strong", "devices": devices, "d_model": 256,
                       "population": 64, "strategy": "lowrank_r1", "how": "A"},
            "trajectory": traj, "guard_precision": "highest",
            "env": _env(xla_flags=flags),
        }
        (tmp_path / f"D{devices}.json").write_text(json.dumps(row))

    assert check.main(["--results", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "MIXED ENV" in out and "xla_flags" in out


def test_identical_environments_still_compare_normally(tmp_path, capsys):
    """The comparability check must not swallow the real comparison."""
    traj = {"digest": "a", "norm": 1.0, "probe": [1.0, 2.0]}
    for devices in (1, 2):
        row = {
            "config": {"mode": "strong", "devices": devices, "d_model": 256,
                       "population": 64, "strategy": "lowrank_r1", "how": "A"},
            "trajectory": traj, "guard_precision": "highest", "env": _env(),
        }
        (tmp_path / f"D{devices}.json").write_text(json.dumps(row))

    assert check.main(["--results", str(tmp_path)]) == 0
    assert "MIXED ENV" not in capsys.readouterr().out


def test_the_guard_gets_its_own_compilation_at_highest_precision():
    """`measure` compiles `generation` under the timing precision, then calls it again
    inside `default_matmul_precision("highest")` for the trajectory fingerprint. That only
    guards anything if the context can still affect an already-compiled function.

    It can: matmul precision is part of jit's cache key, so the second context retraces.
    This test exists because the opposite is entirely plausible (jit caches on the function
    object, which is why fresh lambdas and bound methods miss the cache), and if JAX ever
    stopped keying on precision the guard would quietly start running at the timing
    precision with nothing failing.
    """
    import jax
    import jax.numpy as jnp

    traces = []

    @jax.jit
    def f(x):
        traces.append(1)  # trace time only, not call time
        return x @ x

    x = jnp.eye(4)
    with jax.default_matmul_precision("bfloat16"):
        f(x)
    with jax.default_matmul_precision("highest"):
        f(x)

    assert len(traces) == 2, (
        "entering a different matmul-precision context did not retrace, so run.py's guard "
        "runs at the timing precision rather than at highest"
    )


def test_the_results_dir_from_the_config_is_not_counted_as_a_dirty_worktree(tmp_path):
    """A sweep's own results must not make it stamp itself unreproducible.

    `OUTPUTS` was a hardcoded tuple naming `results` and `results-rehearsal`, so a config
    writing to `results-calibration` had its own output counted as foreign untracked files.
    Every one of the 64 calibration records came back `dirty_worktree: true` for that reason,
    which is exactly the failure `harness.worktree_is_dirty` was written to avoid.
    """
    import harness

    status = "?? experiments/phase2/results-calibration/a.json\n"

    def fake_git(*args):
        if args[0] == "rev-parse" and "--show-toplevel" in args:
            return str(tmp_path)
        if args[0] == "status":
            return status
        return "deadbeef"

    here = tmp_path / "experiments" / "phase2"
    here.mkdir(parents=True)

    assert harness.worktree_is_dirty(here, run.OUTPUTS, fake_git), "hardcoded list misses it"
    assert not harness.worktree_is_dirty(
        here, (*run.OUTPUTS, "results-calibration"), fake_git
    ), "the config's own results_dir must be exempt"


def test_an_output_name_does_not_exempt_files_that_merely_start_with_it(tmp_path):
    """`env.json.bak` is not `env.json`, and a sibling of an output directory is not inside it.

    The filter was `path.startswith(output)`, which exempted both. Provenance failing open
    like that is worse than not recording it: a stray uncommitted file is precisely what
    makes a result unreproducible, and it was being hidden by a name collision.
    """
    import harness

    here = tmp_path / "experiments" / "phase2"
    here.mkdir(parents=True)

    def status_of(path):
        def fake_git(*args):
            if args[0] == "rev-parse" and "--show-toplevel" in args:
                return str(tmp_path)
            if args[0] == "status":
                return f"?? {path}\n"
            return "deadbeef"
        return fake_git

    outputs = ("results", "env.json")
    inside = "experiments/phase2/results/a.json"
    assert not harness.worktree_is_dirty(here, outputs, status_of(inside)), (
        "a file genuinely inside an output directory must stay exempt"
    )
    for stray in ("experiments/phase2/env.json.bak",
                  "experiments/phase2/results-calibration/a.json",
                  "experiments/phase2/results.txt"):
        assert harness.worktree_is_dirty(here, outputs, status_of(stray)), (
            f"{stray} is not one of {outputs} and must count as dirty"
        )


# ------------------------------------------------------------------- the noise floor

noisefloor = _load("noisefloor")


def test_separation_is_reported_in_ulp_not_absolute_units():
    """A gap only means something next to the resolution of the numbers it separates.

    The same 1e-6 gap is 4 ulp on a loss of 2.5 and comfortable on a loss of 0.001. Reporting
    absolute gaps would have made the failing configuration look fine.
    """
    f = np.array([1.0, 1.0 + 2e-7, 5.0], dtype=np.float32)
    s = noisefloor.separation(f)
    assert s["min_gap_ulp"] == pytest.approx(s["min_gap"] / s["ulp"])
    assert s["ulp"] == pytest.approx(float(np.spacing(np.float32(np.mean(np.abs(f))))))


def test_exact_ties_are_counted_and_are_not_treated_as_safe():
    """A tie is the worst case, not a neutral one.

    `centered_ranks` gives tied members different weights, picked by the sort's tie-break, so
    a pair that ties on one backend and differs by an ulp on another can order either way.
    Scoring a zero gap as "maximally close" is the point.
    """
    f = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    s = noisefloor.separation(f)
    assert s["ties"] == 1
    assert s["min_gap_ulp"] == 0.0


def test_a_configuration_below_the_margin_fails(tmp_path, monkeypatch, capsys):
    """The check has to fail, not warn. A warning in a 256-config sweep is not read."""
    monkeypatch.setattr(noisefloor, "scores",
                        lambda *a, **k: np.array([1.0, 1.0, 2.0], dtype=np.float32))
    cfg = {"modes": ["strong"], "devices": [1], "d_model": [256], "population": [4],
           "population_per_device": [4], "strategies": ["lowrank_r1"], "how": ["A"]}
    (tmp_path / "c.yaml").write_text(json.dumps(cfg))
    assert noisefloor.main(["--config", str(tmp_path / "c.yaml"), "--seeds", "1"]) == 1
    assert "TOO CLOSE" in capsys.readouterr().out


def test_a_well_separated_configuration_passes(tmp_path, monkeypatch):
    """And it must not fail everything, or it will be switched off."""
    monkeypatch.setattr(noisefloor, "scores",
                        lambda *a, **k: np.array([1.0, 2.0, 3.0], dtype=np.float32))
    cfg = {"modes": ["strong"], "devices": [1], "d_model": [256], "population": [3],
           "population_per_device": [3], "strategies": ["lowrank_r1"], "how": ["A"]}
    (tmp_path / "c.yaml").write_text(json.dumps(cfg))
    assert noisefloor.main(["--config", str(tmp_path / "c.yaml"), "--seeds", "1"]) == 0


def test_the_worst_seed_is_reported_not_the_first(tmp_path, monkeypatch, capsys):
    """One draw is a sample of the configuration, not a property of it. A configuration that
    is fine on seed 0 and degenerate on seed 1 is not a configuration to benchmark."""
    draws = iter([np.array([1.0, 2.0, 3.0], dtype=np.float32),      # seed 0: wide
                  np.array([1.0, 1.0, 3.0], dtype=np.float32)])     # seed 1: tied
    monkeypatch.setattr(noisefloor, "scores", lambda *a, **k: next(draws))
    cfg = {"modes": ["strong"], "devices": [1], "d_model": [256], "population": [3],
           "population_per_device": [3], "strategies": ["lowrank_r1"], "how": ["A"]}
    (tmp_path / "c.yaml").write_text(json.dumps(cfg))
    assert noisefloor.main(["--config", str(tmp_path / "c.yaml"), "--seeds", "2"]) == 1


def _floor_row(devices, how="A", digest="a", ranks="r", shaping="centered_ranks",
               probe=(1.0, 2.0)):
    """A strong-scaling row aimed at the A-exactness branch of the guard."""
    row = {
        "config": {"mode": "strong", "devices": devices, "d_model": 512,
                   "population": 1024, "strategy": "lowrank_r1", "how": how},
        "trajectory": {"digest": digest, "norm": 1.0, "probe": list(probe)},
        "guard_precision": "highest", "env": _env(),
    }
    if ranks is not None:
        row["ranks"] = {"digest": ranks, "n": 1024, "ties": 0}
    if shaping is not None:
        row["shaping"] = shaping
    return row


def _write_floor_rows(tmp_path, rows):
    for i, row in enumerate(rows):
        (tmp_path / f"{i}.json").write_text(json.dumps(row))


def test_an_a_failure_with_unchanged_ranks_is_a_contraction_bug(tmp_path, capsys):
    """The alarm the guard exists for. Under a rank shaping the update is a function of the
    ordering alone, so the same ordering producing a different update means the contraction
    differs, and no property of the population explains it."""
    _write_floor_rows(tmp_path, [_floor_row(1, digest="a"),
                      _floor_row(2, digest="b", probe=(1.0, 2.0001))])

    assert check.main(["--results", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "contraction itself differs" in out
    assert "NOISE FLOOR" not in out


def test_an_a_failure_with_moved_ranks_is_the_noise_floor(tmp_path, capsys):
    """The other half of the split, and the one measured on 8x A100 at d=512 N=1024: the
    population reordered, so the update changed. Still exit 1, because no scaling number may
    be quoted from it, but it is a fact about the configuration rather than a bug."""
    _write_floor_rows(tmp_path, [_floor_row(1, digest="a", ranks="r1"),
                      _floor_row(2, digest="b", ranks="r2", probe=(1.0, 2.0001))])

    assert check.main(["--results", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "NOISE FLOOR" in out and "noisefloor.py" in out
    assert "contraction itself differs" not in out


def test_an_a_failure_without_a_rank_digest_says_to_re_run(tmp_path, capsys):
    """Results predating the digest are still checked, and the guard says what it cannot
    tell rather than guessing. Invalidating good results would cost a rented node."""
    _write_floor_rows(tmp_path, [_floor_row(1, digest="a", ranks=None, shaping=None),
                      _floor_row(2, digest="b", ranks=None, shaping=None,
                                 probe=(1.0, 2.0001))])

    assert check.main(["--results", str(tmp_path)]) == 1
    assert "no rank digest" in capsys.readouterr().out


def test_a_non_rank_shaping_is_not_held_to_bitwise_equality(tmp_path, capsys):
    """`centered` passes the fitness through to the update, so the sub-rank differences that
    device count introduces reach it and A is no more exact than B. Holding it to exact
    equality would fail every group for a reason that is not a bug."""
    _write_floor_rows(tmp_path, [_floor_row(1, digest="a", shaping="centered"),
                      _floor_row(2, digest="b", shaping="centered",
                                 probe=(1.0, 2.00001))])

    assert check.main(["--results", str(tmp_path)]) == 0


def test_a_non_rank_shaping_still_fails_beyond_rtol(tmp_path, capsys):
    """Relaxing A to B's standard is not relaxing it to no standard."""
    _write_floor_rows(tmp_path, [_floor_row(1, digest="a", shaping="centered"),
                      _floor_row(2, digest="b", shaping="centered", probe=(1.0, 2.1))])

    assert check.main(["--results", str(tmp_path)]) == 1
    assert "exceeds rtol" in capsys.readouterr().out


def test_two_device_counts_shaped_differently_are_not_comparable(tmp_path, capsys):
    """A different shaping is a different computation, so it belongs with the backend and
    the commit rather than as something the numeric comparison tries to absorb."""
    _write_floor_rows(tmp_path, [_floor_row(1, shaping="centered_ranks"),
                      _floor_row(2, shaping="centered")])

    assert check.main(["--results", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "MIXED ENV" in out and "shaping" in out


def test_rank_digest_depends_on_the_ordering_and_not_on_the_values():
    """The whole point: it has to see what `centered_ranks` sees."""
    import numpy as np

    assert (run.rank_digest(np.array([1.0, 2.0, 3.0]))["digest"]
            == run.rank_digest(np.array([10.0, 200.0, 3000.0]))["digest"])
    assert (run.rank_digest(np.array([1.0, 2.0, 3.0]))["digest"]
            != run.rank_digest(np.array([1.0, 3.0, 2.0]))["digest"])


def test_rank_digest_gives_tied_members_one_rank():
    """Midranks, not `argsort(argsort(f))`. Exactly tied members share a weight since the
    midrank change, so permuting them does not change the update and must not read as a
    reordering. `[1, 1, 2]` has to match itself whichever tie the sort happened to put first,
    which `argsort(argsort(f))` does not guarantee."""
    import numpy as np

    assert run.rank_digest(np.array([1.0, 1.0, 2.0]))["ties"] == 1

    # The case that separates the two. `argsort(argsort(f))` gives BOTH of these `[0, 1, 2]`,
    # so it calls them the same ordering; they are not, and `centered_ranks` weights them
    # differently. Midranks give [1.5, 1.5, 3] and [1, 2.5, 2.5].
    assert (run.rank_digest(np.array([1.0, 1.0, 2.0]))["digest"]
            != run.rank_digest(np.array([1.0, 2.0, 2.0]))["digest"])

    # Where the tie sits is what matters, not the values it takes.
    assert (run.rank_digest(np.array([1.0, 1.0, 2.0]))["digest"]
            == run.rank_digest(np.array([0.5, 0.5, 9.0]))["digest"])


# --------------------------------------------------------------------------------------
# M5. comms.py parses HLO text, so the parser is the thing that can silently lie.
# --------------------------------------------------------------------------------------



def test_payload_bytes_sums_a_tuple_shape():
    """A psum over a pytree compiles to one tuple-shaped collective, so the payload is the
    sum of its elements. A `(\\S+)` shape pattern stops at the first comma and reports
    nothing, which read as "strategy B performs no collective at all"."""
    assert comms.payload_bytes("f32[32,32]{1,0}") == 32 * 32 * 4
    assert comms.payload_bytes("(f32[32,32]{1,0}, f32[8]{0})") == 32 * 32 * 4 + 8 * 4
    # Rank zero is one element, not zero.
    assert comms.payload_bytes("f32[]") == 4


def test_an_async_collective_is_counted_once_at_its_result_size():
    """The bug a GPU found and CPU could not.

    XLA:GPU splits a collective into `-start`/`-done`, and the `-start` output is a nested
    tuple of *(operand, result)*. Summing its shapes counts the buffer twice: measured
    exactly 2.00x the prediction for every strategy B configuration, and 1.50x for A, where
    the all-gather operand is `N/D` and the result is `N`. The `-done` output is the result
    alone, so that is what gets counted.
    """
    hlo = (
        "%all-reduce-start = ((f32[1572864]{0}), f32[1572864]{0}) "
        "all-reduce-start(%wrapped_concatenate), channel_id=1\n"
        "%all-reduce-done = f32[1572864]{0} all-reduce-done(%all-reduce-start)\n"
    )
    assert comms.collective_bytes(hlo) == {"all-reduce": 1572864 * 4}


def test_an_unsuffixed_collective_is_counted():
    """XLA:CPU emits the plain opcode with only the result shape."""
    hlo = "%all-reduce = (f32[32,32]{1,0}, f32[32,32]{1,0}) all-reduce(%x), replica_groups={}\n"
    assert comms.collective_bytes(hlo) == {"all-reduce": 2 * 32 * 32 * 4}


def test_a_get_tuple_element_off_a_collective_is_not_a_collective():
    """Every element of a tuple-valued psum is read back by name, so the HLO mentions
    `all-reduce` once per leaf. Counting those would multiply the figure by the leaf count."""
    hlo = (
        "%all-reduce = (f32[4]{0}) all-reduce(%x)\n"
        "%get-tuple-element.1 = f32[4]{0} get-tuple-element(%all-reduce), index=0\n"
        "%get-tuple-element.2 = f32[4]{0} get-tuple-element(%all-reduce), index=1\n"
    )
    assert comms.collective_bytes(hlo) == {"all-reduce": 4 * 4}


def test_the_docs_02_prediction_is_what_comms_compares_against():
    """A gathers N fitness scalars, B all-reduces one params-sized array (docs/02 C1.3).
    Pinned because the whole table is a ratio against these two lines."""
    assert comms.predicted("A", population=256, d_model=512) == 4 * 256
    assert comms.predicted("B", population=256, d_model=512) == comms.params_bytes(512)
    # B does not depend on the population, A does not depend on the model.
    assert comms.predicted("B", population=1024, d_model=512) == comms.predicted(
        "B", population=16, d_model=512)
    assert comms.predicted("A", population=256, d_model=2048) == comms.predicted(
        "A", population=256, d_model=512)


def test_flops_of_scales_with_the_population():
    """Validates the instrument, not the finding.

    `profile.py` concludes that per-device eval FLOPs do not fall with `D`, which is a
    statement about the library and will stop being true when the evaluation distributes.
    Pinning *that* would lock the defect in. What must stay true is that
    `cost_analysis().flops` is a faithful measure of the compiled program, and the check for
    that is proportionality to `N`: doubling the population doubles the work.
    """
    import jax

    from shardes import sharding
    from shardes.core import ShardedES
    from shardes.problems import transformer_block

    def eval_flops(population):
        mesh = sharding.make_mesh(1)
        key = jax.random.key(0)
        params = transformer_block.init(key, d_model=16)
        data = transformer_block.make_batch(
            jax.random.fold_in(key, 1), d_model=16, batch=2, seq=4)
        es = ShardedES(run.STRATEGIES["iid_gaussian"](), n=population, sigma=run.SIGMA,
                       lr=run.LR, mesh=mesh, how="A")

        def ev(state):
            pert, scaled = es.ask(state)
            return es.apply(transformer_block.loss, scaled, pert)(data)

        return profile.flops_of(ev, es.init(key, params))

    small, large = eval_flops(8), eval_flops(16)
    assert small > 0, "cost_analysis reported no flops; the instrument is not measuring"
    assert abs(large / small - 2.0) < 0.05, (
        f"doubling the population changed flops by {large / small:.3f}x, not 2x, so "
        "cost_analysis is not tracking the work"
    )
