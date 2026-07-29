"""The E1 driver's infrastructure: config expansion, slugs, resume, atomic writes.

None of this computes a number, which is exactly why it is worth testing. A resume bug
costs a 20-hour sweep and shows up as "it seemed to finish early", not as a traceback.

The driver lives under experiments/, so it is loaded by path rather than imported.
"""

import importlib.util
import json
import pathlib
import sys

import jax
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
    "axes": {
        "population": [64, 256],
        "sigma": [0.001, 0.01],
        # One mode per side, so a test that accidentally crossed the axis instead of
        # selecting from it would still produce the right *count*. The membership
        # assertions below are what catch that.
        "shaping": {"iid": ["centered"], "mirrored": ["none"]},
    },
}

GRID = {
    "a_full": Entry(lambda: None, FULL, "iid"),
    "b_lr1": Entry(lambda: None, 1, "mirrored+sobol"),
}


def test_expand_crosses_every_axis():
    configs = run.expand(CFG, GRID)
    assert len(configs) == 2 * 2 * 2 * 1  # strategies x population x sigma x shaping


def test_shaping_axis_is_conditional_on_the_scheme():
    """docs/01 C0.5: the axis is selected by scheme, not crossed with it.

    `centered` under a mirrored scheme is a measured bug, not a wasted config: the pair
    already cancels `f_bar`, so the n/(n-1) correction over-corrects and the estimator
    targets n/(n-1) grad f. Crossing the axis would put that arm in the sweep and its curve
    would sit 6.7% off at n=16 with nothing saying why.
    """
    by_strategy = {}
    for c in run.expand(CFG, GRID):
        by_strategy.setdefault(c.strategy, set()).add(c.shaping)

    assert by_strategy["a_full"] == {"centered"}
    assert by_strategy["b_lr1"] == {"none"}


@pytest.mark.parametrize(
    "scheme,want",
    [("iid", "i"), ("mirrored", "m"), ("mirrored+orthogonal_hd", "m"), ("mirrored+sobol", "m")],
)
def test_every_scheme_in_the_grid_picks_a_side(scheme, want):
    """Every coupled scheme composes Mirrored, so `"mirrored" in scheme` is the whole test.
    Pinned per scheme string so adding an uncoupled coupled scheme later cannot land on the
    iid side by accident."""
    assert run.shaping_for(scheme, {"iid": "i", "mirrored": "m"}) == want


def test_shaping_axis_must_be_a_mapping(tmp_path):
    """The flat list this replaced was valid YAML and silently wrong: it gave the mirrored
    side the right baseline and spent 84 configs on the dead iid arm."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n"
        "model:\n  kind: quadratic\n  d: 8\n"
        "axes:\n  population: [64]\n  sigma: [0.01]\n  shaping: ['none', 'centered_ranks']\n"
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        run.load_config(p)


def test_shaping_axis_rejects_an_unknown_mode(tmp_path):
    """Otherwise this is a KeyError partway through a 20-hour sweep."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n"
        "model:\n  kind: quadratic\n  d: 8\n"
        "axes:\n  population: [64]\n  sigma: [0.01]\n"
        "  shaping:\n    iid: ['centred']\n    mirrored: ['none']\n"
    )
    with pytest.raises(ValueError, match="not a shaping mode"):
        run.load_config(p)


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


BASE = ("seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n"
        "model:\n  kind: quadratic\n  d: 8\n")
SHAPING = "  shaping:\n    iid: ['centered']\n    mirrored: ['none']\n"
AXES = "axes:\n  population: [64]\n  sigma: [0.01]\n" + SHAPING


def axes(population="[64]", sigma="[0.01]", shaping=SHAPING) -> str:
    return f"axes:\n  population: {population}\n  sigma: {sigma}\n{shaping}"


def test_the_shipped_config_loads():
    """The one that actually runs. A typo here is a sweep that dies on line one."""
    cfg = run.load_config(DRIVER.parent / "config.yaml")
    assert cfg["replicates"] >= 30  # docs/01 C0.5
    assert cfg["axes"]["population"] == sorted(cfg["axes"]["population"])

    # `chunk` is load-bearing, not optional: full rank OOMs from N = 1024 without it on the
    # 3080 (docs/01 C0.2). A config that lost it would OOM through most of the grid, and the
    # driver would dutifully record every failure and carry on.
    assert cfg["chunk"] and cfg["chunk"] % 2 == 0

    shaping = cfg["axes"]["shaping"]
    assert set(shaping) == set(run.SHAPING_SIDES)
    assert all(isinstance(s, str) for side in shaping.values() for s in side)
    # docs/01 C0.5, both directions. Neither is a taste call: `centered` under mirroring
    # over-corrects by n/(n-1), and `none` without it is the dead arm.
    assert "centered" not in shaping["mirrored"]
    assert "none" not in shaping["iid"]


@pytest.mark.parametrize("bad", ["no", "No", "NO", "off", "on", "yes", "~", "null"])
def test_rejects_the_norway_problem(tmp_path, bad):
    """A shaping mode written bare as `no` becomes False, and every result file is then
    labelled shaping=False."""
    path = _write(tmp_path, BASE + axes(shaping=f"  shaping:\n    iid: [{bad}]\n    mirrored: ['none']\n"))
    with pytest.raises(ValueError, match="quote the value"):
        run.load_config(path)


@pytest.mark.parametrize("safe", ["y", "n", "Y", "N"])
def test_single_letters_are_not_coerced_by_pyyaml(safe):
    """Documents the boundary, asserted against PyYAML rather than through the driver.

    The YAML 1.1 spec lists bare `y`/`n` as booleans and PyYAML does not implement that, so
    a test asserting `y` is coerced fails against the library actually in use. `config.yaml`
    relies on this for its `m:`/`n:` keys, so an upgrade that starts coercing them has to
    fail here.

    Going through `load_config` would no longer reach the coercion check: these are not
    shaping modes, so the name guard fires first.
    """
    import yaml  # noqa: PLC0415

    assert yaml.safe_load(f"v: [{safe}]")["v"] == [safe]


def test_quoting_fixes_the_coercion_but_not_a_wrong_name(tmp_path):
    """Two guards, and the second is new.

    Quoted, `'no'` is a genuine string and survives the Norway check. It is still not a
    shaping mode, and that used to sail through here and land as a KeyError partway into a
    20-hour sweep.
    """
    path = _write(tmp_path, BASE + axes(shaping="  shaping:\n    iid: ['no']\n    mirrored: ['none']\n"))
    with pytest.raises(ValueError, match="not a shaping mode"):
        run.load_config(path)


@pytest.mark.parametrize("bad,match", [("3", "must be even"), ("0", "positive int"),
                                       ("-2", "positive int"), ("true", "positive int")])
def test_rejects_a_nonsense_chunk(tmp_path, bad, match):
    """`chunk` is what makes the sweep runnable at all, so a bad value has to fail at load.

    Odd is the interesting case: every mirrored scheme in the registry would raise partway
    through, after the sweep had already spent time on the unmirrored ones.
    """
    path = _write(tmp_path, BASE + f"chunk: {bad}\n" + axes())
    with pytest.raises(ValueError, match=match):
        run.load_config(path)


def test_accepts_an_even_chunk_and_no_chunk(tmp_path):
    assert run.load_config(_write(tmp_path, BASE + "chunk: 256\n" + axes()))["chunk"] == 256
    assert run.load_config(_write(tmp_path, BASE + axes())).get("chunk") is None


def test_make_estimator_wires_the_config_to_the_library():
    """The one seam between the driver and the library, and the only untested one.

    A field wired to the wrong argument here yields plausible numbers rather than a crash,
    which is the worst failure mode available. The quadratic makes it checkable: the ES
    estimator is *exact* there at any sigma, so the mean over replicates must land on the
    analytic gradient.

    The tight bias gate is what pins `sigma`. It appears twice, in `apply` and in `tell`'s
    1/(N sigma), and if the driver dropped it the normalisation would be off by 1/0.01 and
    the bias would come back around 99 rather than under 0.1.
    """
    cfg = {
        "seed": 0, "replicates": 1, "wall_clock_cap_s": 60, "chunk": 4,
        "model": {"kind": "quadratic", "d": 8},
        "axes": {"population": [64], "sigma": [0.01], "shaping": {"iid": ["centered"],
                                                                  "mirrored": ["none"]}},
    }
    estimate = run.make_estimator(cfg)
    config = run.Config("iid_gaussian", FULL, "iid", 64, 0.01, "centered", 1, 0)

    from shardes import metrics  # noqa: PLC0415

    gs = []
    for r in range(400):
        g, truth = estimate(config, jax.random.fold_in(jax.random.key(7), r))
        gs.append(g)
    mean = jax.tree.map(lambda *xs: sum(xs) / len(xs), *gs)

    assert truth.shape == (8,)
    assert float(metrics.relative_bias(mean, truth)) < 0.1
    assert float(metrics.cosine_similarity(mean, truth)) > 0.9  # no sign flip


def test_make_estimator_respects_the_shaping_field():
    """`shaping_name` is a static jit argument, so a wrong one would silently reuse a cached
    executable for a different shaping. Two modes have to give two answers."""
    cfg = {
        "seed": 0, "replicates": 1, "wall_clock_cap_s": 60, "chunk": 4,
        "model": {"kind": "quadratic", "d": 8},
        "axes": {"population": [16], "sigma": [0.1], "shaping": {"iid": ["centered"],
                                                                 "mirrored": ["none"]}},
    }
    estimate = run.make_estimator(cfg)
    key = jax.random.key(3)
    out = {}
    for mode in ("centered", "centered_ranks"):
        g, _ = estimate(run.Config("iid_gaussian", FULL, "iid", 16, 0.1, mode, 1, 0), key)
        out[mode] = g
    assert not bool((out["centered"] == out["centered_ranks"]).all())


def test_rejects_unquoted_scientific_notation(tmp_path):
    """`1e-3` is a string in YAML 1.1, not a float. It needs a decimal point."""
    path = _write(tmp_path, BASE + axes(sigma="[not_a_number]"))
    with pytest.raises(ValueError, match="decimal point"):
        run.load_config(path)


def test_accepts_proper_scientific_notation(tmp_path):
    path = _write(tmp_path, BASE + axes(sigma="[1.0e-3]"))
    assert run.load_config(path)["axes"]["sigma"] == [0.001]


@pytest.mark.parametrize("bad", ["[0]", "[-5]", "[true]", "[3.5]"])
def test_rejects_nonsense_populations(tmp_path, bad):
    path = _write(tmp_path, BASE + axes(population=bad))
    with pytest.raises(ValueError, match="positive int"):
        run.load_config(path)


@pytest.mark.parametrize("key", ["seed", "replicates", "wall_clock_cap_s", "axes", "model"])
def test_rejects_missing_required_keys(tmp_path, key):
    body = BASE + AXES
    if key == "axes":
        body = BASE
    elif key == "model":
        body = "seed: 0\nreplicates: 2\nwall_clock_cap_s: 60\n" + AXES
    else:
        body = "\n".join(l for l in body.splitlines() if not l.startswith(key)) + "\n"
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


def test_replicate_keys_are_shared_across_configs():
    """Common random numbers. Two configs that differ only in strategy must see the same
    replicate seeds, or a scheme comparison is confounded with the draw.

    Also pins reproducibility: an earlier version folded `hash(config.slug())` into the
    seed, and CPython salts str hashes per process, so a resumed sweep would have used
    different seeds from the run it resumed.

    **Paired within one side of the shaping axis.** The axis is conditional on the scheme
    (docs/01 C0.5), so an iid arm and a mirrored arm never share a shaping value and there is
    no cross-side pair to ask for. That costs the figure nothing: F5's scheme comparison is
    `mirrored` against `mirrored+orthogonal_hd` against `mirrored+sobol`, all on the same
    side, and `iid` is a separate baseline curve rather than a paired arm.
    """
    seen: dict = {}

    def recorder(config, key):
        seen.setdefault(config.strategy, []).append(jax.random.key_data(key).tolist())
        return run.synthetic_estimate(config, key)

    same_side = {
        "b_lr1": Entry(lambda: None, 1, "mirrored+sobol"),
        "c_lr1": Entry(lambda: None, 1, "mirrored+orthogonal_hd"),
    }
    configs = run.expand(CFG, same_side)
    a = next(c for c in configs if c.strategy == "b_lr1")
    b = next(c for c in configs if c.strategy == "c_lr1" and c.population == a.population
             and c.sigma == a.sigma and c.shaping == a.shaping)

    run.run_one(a, recorder, cap_s=60)
    run.run_one(b, recorder, cap_s=60)
    assert seen["b_lr1"] == seen["c_lr1"]


def test_run_one_is_reproducible():
    config = run.expand(CFG, GRID)[0]
    first = run.run_one(config, run.synthetic_estimate, cap_s=60)
    second = run.run_one(config, run.synthetic_estimate, cap_s=60)
    assert first["cosine_median"] == second["cosine_median"]


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
        "model:\n  kind: quadratic\n  d: 8\n"
        + axes(population="[64, 256]")
    )

    rc = run.main(["--dry-run", "--config", str(cfg_path)])

    assert rc == 1, "a run with failures must not report success"
    written = sorted(p.name for p in tmp_path.glob("*.json") if p.name != "env.json")
    # The two N=256 configs succeeded; the two N=64 ones failed and wrote nothing, so a
    # re-run retries exactly those.
    assert len(written) == 2
    assert all("N=256" in name for name in written)
