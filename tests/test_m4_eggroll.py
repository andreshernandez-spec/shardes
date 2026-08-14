"""The EGGROLL arm: is it running their code, and is it a fair comparison?

Every test here skips when `hyperscalees` is not installed, which is the normal state.
`CLAUDE.md` keeps it a comparison target rather than a dependency, and it is GPL-3.0 against
this project's Apache-2.0, so it is never vendored and the suite must pass without it.

**These are not throughput tests.** They check the three properties a number from that arm
depends on and that no timing would reveal: that the perturbation is applied at all, that it
is rank 1, and that their scheme is antithetic. If the arm silently stopped perturbing, it
would get faster and the benchmark would report a win.
"""

import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

PHASE2 = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase2"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"phase2_{name}", PHASE2 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"phase2_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


m4 = _load("m4")

from shardes import sharding  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402


def _eggroll():
    got = m4._eggroll_or_reason()
    if got[0] == "unavailable":
        pytest.skip(f"hyperscalees unavailable: {got[1]}")
    return got[0]


@pytest.fixture(scope="module")
def noiser():
    eg = _eggroll()
    params = transformer_block.init(jax.random.key(0), d_model=32)
    frozen, nsp = eg.init_noiser(params, m4.SIGMA, m4.LR, rank=1, noise_reuse=1)
    return eg, frozen, nsp, params["wq"]


def test_do_mm_without_iterinfo_is_exactly_the_unperturbed_matmul(noiser):
    """`iterinfo=None` is their "no member" path. It has to be the base model exactly.

    The arm relies on this to know the perturbation is a correction rather than a
    reparameterisation, and `docs/03` compares against the same base model.
    """
    eg, frozen, nsp, w = noiser
    x = jnp.eye(w.shape[1])
    assert jnp.allclose(eg.do_mm(frozen, nsp, w, jax.random.key(7), None, x), x @ w.T)


def test_the_perturbation_is_applied_and_is_rank_one(noiser):
    """Rank 1 because the arm passes `rank=1` to match `LowRank(r=1)`.

    A silently unperturbed arm is the failure that would flatter the benchmark: it removes
    work, so it shows up as throughput rather than as an error.
    """
    eg, frozen, nsp, w = noiser
    x = jnp.eye(w.shape[1])
    base = eg.do_mm(frozen, nsp, w, jax.random.key(7), None, x)
    member = eg.do_mm(frozen, nsp, w, jax.random.key(7), (0, 0), x)
    assert not jnp.allclose(base, member), "perturbation was not applied"
    assert jnp.linalg.matrix_rank(member - base) == 1


def test_their_scheme_is_antithetic_so_the_matched_arm_is_mirrored(noiser):
    """Members `2k` and `2k+1` are exact opposites: `thread_id // 2` with the sign from `% 2`.

    **This is why `m4.py` pairs the arm against `mirrored_lr1` rather than `lowrank_r1`.**
    `N` EGGROLL members are `N/2` directions, so the unmirrored arm would be doing twice the
    distinct sampling for the same `N` and the difference would be reported as throughput.
    Asserted here rather than trusted from reading their source, because it decides whether
    the comparison in `docs/03` is fair.
    """
    eg, frozen, nsp, w = noiser
    x = jnp.eye(w.shape[1])
    key = jax.random.key(7)
    base = eg.do_mm(frozen, nsp, w, key, None, x)
    plus = eg.do_mm(frozen, nsp, w, key, (0, 0), x) - base
    minus = eg.do_mm(frozen, nsp, w, key, (0, 1), x) - base
    assert jnp.allclose(plus, -minus, atol=1e-6)


def test_noise_reuse_zero_freezes_the_noise_across_generations(noiser):
    """Their default `noise_reuse=0` means reuse forever, not "do not reuse".

    `true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse`. The arm passes 1 so each
    generation resamples. This test exists so that if they change the sentinel, the arm's
    reason for passing 1 fails loudly instead of the benchmark quietly evaluating one fixed
    set of perturbations forever.
    """
    eg, _, _, w = noiser
    params = {"wq": w}
    x = jnp.eye(w.shape[1])
    key = jax.random.key(7)

    frozen0, nsp0 = eg.init_noiser(params, m4.SIGMA, m4.LR, rank=1, noise_reuse=0)
    frozen1, nsp1 = eg.init_noiser(params, m4.SIGMA, m4.LR, rank=1, noise_reuse=1)
    gen0 = [eg.do_mm(frozen0, nsp0, w, key, (e, 0), x) for e in (0, 1)]
    gen1 = [eg.do_mm(frozen1, nsp1, w, key, (e, 0), x) for e in (0, 1)]

    assert jnp.allclose(gen0[0], gen0[1]), "noise_reuse=0 no longer freezes the noise"
    assert not jnp.allclose(gen1[0], gen1[1]), "noise_reuse=1 no longer resamples"


def test_the_arm_descends():
    """End to end: their update actually optimizes this project's transformer block.

    Loose on purpose. This is a wiring check, not a convergence claim: a sign error or a
    mis-shaped `es_map` shows up as a loss that rises or sticks, and that is all it catches.
    """
    _eggroll()
    shape = m4.Shape(d_model=32, population=8, batch=2, seq=4)
    step, state = m4.arm_eggroll(shape, sharding.make_mesh(1))
    batch = transformer_block.make_batch(
        jax.random.fold_in(jax.random.key(m4.SEED), 1), d_model=32, batch=2, seq=4
    )
    first = float(transformer_block.loss(state[1], batch))
    for _ in range(30):
        state = step(state)
    assert float(transformer_block.loss(state[1], batch)) < first


def test_an_odd_population_is_refused_rather_than_silently_unpaired():
    """Odd `N` leaves one member unpaired under `thread_id // 2`, which is not their scheme."""
    _eggroll()
    got = m4.arm_eggroll(m4.Shape(d_model=32, population=7, batch=2, seq=4),
                         sharding.make_mesh(1))
    assert got[0] == "unavailable" and "odd" in got[1]


def test_a_missing_hyperscalees_is_a_reason_not_a_crash():
    """The suite has to pass without it, and `--dry-run` has to say why it is absent."""
    got = m4._eggroll_or_reason()
    assert got[0] != "unavailable" or isinstance(got[1], str) and got[1]


def test_results_record_the_external_revision():
    """A result that pins only this repo's commit pins half the comparison. The A100 rows
    at a858998 were measured against HyperscaleES b77f7d6 and nothing in the file said
    so; this is the regression test for the fix."""
    _eggroll()
    prov = m4._eggroll_provenance()
    assert prov is not None
    assert len(prov["modules_sha256"]) == 16
    # commit is None on a non-git install, a full SHA on the documented editable one.
    assert prov["commit"] is None or len(prov["commit"]) == 40


def test_provenance_is_none_when_the_arm_is_absent():
    """The env field must say 'absent' rather than crash the whole run."""
    if m4._eggroll_or_reason()[0] != "unavailable":
        pytest.skip("hyperscalees installed; the absent path is exercised on CI-like envs")
    assert m4._eggroll_provenance() is None
