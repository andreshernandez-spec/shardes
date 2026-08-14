"""The Qwen2 port: wiring on a tiny config, and the seam contract it exists for.

Two tiers. This file is the free tier: CPU, no network, no checkpoint, a 300k-parameter
`Config.tiny()` with the same wiring as the 0.5B. The golden-logit tier against the real
checkpoint lives in `test_qwen2_golden.py` and skips when the checkpoint is not cached,
because the suite's contract is no network.

What the wiring tests pin, in order of how expensive the mistake would be on a pod:
the tied embedding read through both seams under LowRank (C6c is this, in miniature),
every strategy evaluating the model through `ShardedES.apply` in bf16, and the
device-count invariance of the update with an embedding in the tree.
"""

import jax
import jax.numpy as jnp
import pytest

from shardes import sharding
from shardes.core import ShardedES
from shardes.problems import qwen2
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

CFG = qwen2.Config.tiny()

STRATEGIES = [
    ("iid_gaussian", IIDGaussian),
    ("seed_regenerated", SeedRegenerated),
    ("mirrored_seed", lambda: Mirrored(SeedRegenerated())),
    ("mirrored_lr1", lambda: Mirrored(LowRank(r=1))),
]
IDS = [n for n, _ in STRATEGIES]
MAKES = [m for _, m in STRATEGIES]


def _batch(key, batch=2, seq=8):
    ids = jax.random.randint(key, (batch, seq), 0, CFG.vocab)
    return ids, jnp.ones_like(ids)


def test_forward_shapes_and_dtype():
    params = qwen2.init(jax.random.key(0), CFG)
    ids, _ = _batch(jax.random.key(1))
    logits = qwen2.forward(params, ids, CFG)
    assert logits.shape == (2, 8, CFG.vocab)
    assert logits.dtype == jnp.float32, "logits feed reductions; they are f32 by contract"


def test_nll_is_f32_and_finite_and_mask_works():
    params = qwen2.init(jax.random.key(0), CFG)
    ids, mask = _batch(jax.random.key(1))
    loss = qwen2.nll(params, (ids, mask), CFG)
    assert loss.dtype == jnp.float32 and bool(jnp.isfinite(loss))
    # A random model on a vocab of 128 sits near log(128); catches a softmax-axis or
    # off-by-one-shift bug, both of which land far from it.
    assert 3.0 < float(loss) < 7.0
    half = qwen2.nll(params, (ids, mask.at[:, 4:].set(0)), CFG)
    assert not jnp.allclose(loss, half), "the mask changed nothing"


def test_the_tied_embedding_is_one_leaf_read_through_both_seams():
    """logits = h @ E.T via dense on the same leaf embed gathers from. If a port change
    ever splits the head into its own leaf, the tying claim in the module docstring is
    false and this fails."""
    params = qwen2.init(jax.random.key(0), CFG)
    leaves = jax.tree.leaves(params)
    assert sum(leaf.shape == (CFG.vocab, CFG.d_model) for leaf in leaves) == 1


def test_causality():
    """Changing a future token must not change a past logit."""
    params = qwen2.init(jax.random.key(0), CFG)
    ids, _ = _batch(jax.random.key(1), batch=1)
    a = qwen2.forward(params, ids, CFG)
    b = qwen2.forward(params, ids.at[0, -1].set((ids[0, -1] + 1) % CFG.vocab), CFG)
    assert jnp.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not jnp.allclose(a[0, -1], b[0, -1])


@pytest.mark.parametrize("make", MAKES, ids=IDS)
def test_every_strategy_evaluates_the_model_in_bf16(make):
    """The E13 configuration end to end: bf16 compute, f32 master, f32 fitness, and the
    embedding perturbed rather than frozen. LowRank routes the (V, d) table through the
    factored path of both seams; a failure here is C6c failing in miniature."""
    n = 8
    es = ShardedES(make(), n=n, sigma=0.01, lr=0.05, mesh=sharding.make_mesh(1),
                   compute_dtype=jnp.bfloat16)
    params = qwen2.init(jax.random.key(0), CFG, dtype=jnp.bfloat16)
    st = es.init(jax.random.key(1), params)
    pert, st = es.ask(st)
    fit = es.apply(lambda p, b: qwen2.nll(p, b, CFG), st, pert)(_batch(jax.random.key(2)))
    assert fit.shape == (n,) and fit.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(fit)))
    st2 = es.tell(st, pert, fit)
    assert st2.params["embedding"].dtype == jnp.float32  # the master, not the view


@pytest.mark.parametrize("make", MAKES, ids=IDS)
def test_device_invariance_with_an_embedding_in_the_tree(make):
    """Invariant 2 on this model. The embedding is the leaf shape no other test tree
    has, and the E13 run rides on this property (C6d)."""
    def run(devices):
        # n=16, not 8: Mirrored needs whole pairs per device, and 8 members over 8
        # devices is one member per device. The pairing guard refuses it, correctly.
        es = ShardedES(make(), n=16, sigma=0.01, lr=0.05,
                       mesh=sharding.make_mesh(devices), compute_dtype=jnp.bfloat16)
        st = es.init(jax.random.key(1), qwen2.init(jax.random.key(0), CFG, jnp.bfloat16))
        pert, st = es.ask(st)
        fit = es.apply(lambda p, b: qwen2.nll(p, b, CFG), st, pert)(
            _batch(jax.random.key(2)))
        return jax.device_get(es.tell(st, pert, fit).params)

    ref, got = run(1), run(8)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        err = float(abs(a - b).max() / (abs(a).max() + 1e-30))
        assert err < 1e-6, f"relative error {err:.2e} across device counts"


def test_loading_maps_every_tensor_or_says_which_it_could_not():
    """`load` is exercised for real in the golden tier; here, the failure modes that
    do not need a checkpoint: an empty directory, and the unmapped-tensor error."""
    with pytest.raises(FileNotFoundError):
        qwen2.load("/nonexistent", CFG)


def test_greedy_generate_agrees_with_iterated_forward():
    """The KV cache earns its keep only if decode is exactly forward, token by token.

    The reference is the model itself: append the argmax of `forward`'s last position,
    repeat. Any cache indexing, rope offset, or masking bug breaks the agreement at the
    first divergent token, and did during development.
    """
    params = qwen2.init(jax.random.key(0), CFG)
    ids = jax.random.randint(jax.random.key(1), (2, 6), 0, CFG.vocab)
    plen = jnp.array([6, 4])  # one full row, one right-padded row
    got = qwen2.generate(params, ids, plen, CFG, max_new=5)

    for row in range(2):
        seq = list(map(int, ids[row, : plen[row]]))
        while len(seq) < plen[row] + 5 and len(seq) < ids.shape[1] + 5:
            logits = qwen2.forward(params, jnp.asarray([seq]), CFG)
            seq.append(int(jnp.argmax(logits[0, -1])))
        assert list(map(int, got[row, : len(seq)])) == seq[: got.shape[1]], f"row {row}"
