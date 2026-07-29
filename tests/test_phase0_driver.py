"""The E1 driver's infrastructure: config expansion, slugs, resume, atomic writes.

None of this computes a number, which is exactly why it is worth testing. A resume bug
costs a 20-hour sweep and shows up as "it seemed to finish early", not as a traceback.

The driver lives under experiments/, so it is loaded by path rather than imported.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

from shardes.strategies.registry import FULL, Entry, check_entry

DRIVER = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase0" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase0_run", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so a module
    # loaded by path has to be registered before exec_module or it fails on `Config`.
    sys.modules["phase0_run"] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()


CFG = {
    "seed": 0,
    "replicates": 5,
    "wall_clock_cap_s": 60,
    "axes": {"population": [64, 256], "sigma": [0.001, 0.01], "shaping": ["none"]},
}

GRID = {
    "a_full": Entry(lambda: None, FULL, "iid"),
    "b_lr1": Entry(lambda: None, 1, "mirrored+sobol"),
}


def test_expand_crosses_every_axis():
    configs = run.expand(CFG, GRID)
    assert len(configs) == 2 * 2 * 2 * 1  # strategies x population x sigma x shaping


def test_expand_is_deterministic():
    """Result filenames come from this order, so a reshuffle orphans earlier results."""
    assert run.expand(CFG, GRID) == run.expand(CFG, GRID)


def test_expand_orders_by_population():
    """A truncated sweep should leave complete small-N curves, not a ragged edge."""
    pops = [c.population for c in run.expand(CFG, GRID) if c.strategy == "a_full"]
    assert pops == sorted(pops)


def test_expand_validates_against_the_sweep_grid():
    """docs/01 C0.5: full-rank sobol is not constructible, so it must not reach a run."""
    bad = {"oops": Entry(lambda: None, FULL, "mirrored+sobol")}
    with pytest.raises(ValueError, match="sobol is low-rank only"):
        run.expand(CFG, bad)


def test_dry_run_grid_is_itself_valid():
    """The rehearsal grid is not exempt from the constraint it is rehearsing."""
    for name, entry in run.DRY_RUN_GRID.items():
        check_entry(name, entry)


def test_slug_is_stable_and_filesystem_safe():
    c = run.expand(CFG, GRID)[0]
    slug = c.slug()
    assert slug == c.slug()
    assert not (set(slug) & set("/\\ :*?\"<>|"))


def test_slug_formats_sigma_consistently():
    """0.001 must not be '0.001' on one platform and '1e-03' on another, or a resumed
    sweep re-runs everything and silently doubles the cost."""
    configs = {c.slug() for c in run.expand(CFG, GRID)}
    assert all("sigma=1e-03" in s or "sigma=1e-02" in s for s in configs)


def test_distinct_configs_get_distinct_slugs():
    configs = run.expand(CFG, GRID)
    assert len({c.slug() for c in configs}) == len(configs)


def test_write_atomic_leaves_no_partial_file(tmp_path):
    target = tmp_path / "r.json"
    run.write_atomic(target, {"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_write_atomic_overwrites_cleanly(tmp_path):
    target = tmp_path / "r.json"
    run.write_atomic(target, {"x": 1})
    run.write_atomic(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 2}


def test_resume_skips_existing_results(tmp_path, monkeypatch):
    """The property the whole driver exists for."""
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    configs = run.expand(CFG, GRID)
    assert all(not c.path().exists() for c in configs)

    run.write_atomic(configs[0].path(), {"done": True})
    outstanding = [c for c in configs if not c.path().exists()]
    assert len(outstanding) == len(configs) - 1
    assert configs[0] not in outstanding


def _write(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


BASE = "seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n"
AXES = "axes:\n  population: [64]\n  sigma: [0.01]\n  shaping: ['none']\n"


def test_the_shipped_config_loads():
    """The one that actually runs. A typo here is a sweep that dies on line one."""
    cfg = run.load_config(DRIVER.parent / "config.yaml")
    assert cfg["replicates"] >= 30  # docs/01 C0.5
    assert cfg["axes"]["population"] == sorted(cfg["axes"]["population"])
    assert all(isinstance(s, str) for s in cfg["axes"]["shaping"])


@pytest.mark.parametrize("bad", ["no", "No", "NO", "off", "on", "yes", "~", "null"])
def test_rejects_the_norway_problem(tmp_path, bad):
    """A shaping mode written bare as `no` becomes False, and every result file is then
    labelled shaping=False.

    The exact token set is PyYAML's, checked rather than taken from the YAML 1.1 spec:
    the spec lists bare `y`/`n` as booleans and PyYAML does not implement that, so a test
    asserting `y` is coerced fails against the library actually in use.
    """
    path = _write(tmp_path, BASE + f"axes:\n  population: [64]\n  sigma: [0.01]\n  shaping: [{bad}]\n")
    with pytest.raises(ValueError, match="quote the value"):
        run.load_config(path)


@pytest.mark.parametrize("safe", ["y", "n", "Y", "N"])
def test_single_letters_are_not_coerced_by_pyyaml(tmp_path, safe):
    """Documents the boundary. If a PyYAML upgrade starts coercing these, this test
    fails and the config comment needs revisiting."""
    path = _write(tmp_path, BASE + f"axes:\n  population: [64]\n  sigma: [0.01]\n  shaping: [{safe}]\n")
    assert run.load_config(path)["axes"]["shaping"] == [safe]


def test_accepts_quoted_shaping_names(tmp_path):
    path = _write(tmp_path, BASE + "axes:\n  population: [64]\n  sigma: [0.01]\n  shaping: ['no', 'none']\n")
    assert run.load_config(path)["axes"]["shaping"] == ["no", "none"]


def test_rejects_unquoted_scientific_notation(tmp_path):
    """`1e-3` is a string in YAML 1.1, not a float. It needs a decimal point."""
    path = _write(tmp_path, BASE + "axes:\n  population: [64]\n  sigma: [not_a_number]\n  shaping: ['none']\n")
    with pytest.raises(ValueError, match="decimal point"):
        run.load_config(path)


def test_accepts_proper_scientific_notation(tmp_path):
    path = _write(tmp_path, BASE + "axes:\n  population: [64]\n  sigma: [1.0e-3]\n  shaping: ['none']\n")
    assert run.load_config(path)["axes"]["sigma"] == [0.001]


@pytest.mark.parametrize("bad", ["[0]", "[-5]", "[true]", "[3.5]"])
def test_rejects_nonsense_populations(tmp_path, bad):
    path = _write(tmp_path, BASE + f"axes:\n  population: {bad}\n  sigma: [0.01]\n  shaping: ['none']\n")
    with pytest.raises(ValueError, match="positive int"):
        run.load_config(path)


@pytest.mark.parametrize("key", ["seed", "replicates", "wall_clock_cap_s", "axes"])
def test_rejects_missing_required_keys(tmp_path, key):
    body = BASE + AXES
    body = "\n".join(l for l in body.splitlines() if not l.startswith(key)) + "\n"
    if key == "axes":
        body = BASE
    with pytest.raises(ValueError, match="missing required key"):
        run.load_config(_write(tmp_path, body))


def test_capture_env_records_what_a_paper_needs():
    env = run.capture_env()
    for key in ("commit", "dirty_worktree", "jax", "device_kind", "device_platform",
                "platform", "python", "xla_flags"):
        assert key in env, key
    assert isinstance(env["dirty_worktree"], bool)


def test_run_one_aggregates_over_replicates():
    """Median and IQR, never a single number (docs/conventions.md)."""
    config = run.expand(CFG, GRID)[0]
    rec = run.run_one(config, run.synthetic_estimate, cap_s=60)
    assert rec["replicates_completed"] == CFG["replicates"]
    assert not rec["truncated"]
    assert rec["cosine_q1"] <= rec["cosine_median"] <= rec["cosine_q3"]
    assert rec["relative_mse_median"] >= 0.0


def test_run_one_truncates_on_the_wall_clock_cap():
    """Exceeding the cap stops that config rather than stalling the sweep."""
    config = run.expand(CFG, GRID)[0]
    rec = run.run_one(config, run.synthetic_estimate, cap_s=0.0)
    assert rec["truncated"]
    assert rec["replicates_completed"] == 1


def test_a_zero_cap_still_yields_one_replicate():
    """Checking the cap before a replicate instead of after gives a config with no data
    at all, which is worse than overrunning. Guard against that regression."""
    config = run.expand(CFG, GRID)[0]
    for cap in (-1.0, 0.0):
        rec = run.run_one(config, run.synthetic_estimate, cap_s=cap)
        assert rec["replicates_completed"] >= 1
        assert rec["truncated"]


def test_a_failing_config_does_not_kill_the_sweep(tmp_path, monkeypatch, capsys):
    """One bad config must cost one config, not the session."""
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "HERE", tmp_path)
    monkeypatch.setattr(run, "STRATEGIES", GRID)

    real_estimate = run.synthetic_estimate  # capture before patching, or flaky recurses

    def flaky(config, key):
        if config.population == 64:
            raise RuntimeError("simulated OOM")
        return real_estimate(config, key)

    monkeypatch.setattr(run, "synthetic_estimate", flaky)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n"
        "axes:\n  population: [64, 256]\n  sigma: [0.01]\n  shaping: ['none']\n"
    )

    rc = run.main(["--dry-run", "--config", str(cfg_path)])

    assert rc == 1, "a run with failures must not report success"
    written = sorted(p.name for p in tmp_path.glob("*.json") if p.name != "env.json")
    # The two N=256 configs succeeded; the two N=64 ones failed and wrote nothing, so a
    # re-run retries exactly those.
    assert len(written) == 2
    assert all("N=256" in name for name in written)
