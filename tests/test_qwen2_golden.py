"""The Qwen2 port against the released checkpoint: same weights, same logits.

The reference is Hugging Face `transformers` running the same checkpoint on CPU,
computed live rather than committed: a golden file of usable size would hold a
truncated summary, and the transformers call *is* the committed, re-runnable script.

Everything here skips unless the validation stack is present, because the suite makes
no network calls and torch is not a dependency of anything: the checkpoint must already
be in the HF cache (`experiments/countdown/fetch.py` downloads it) and `torch` +
`transformers` must be importable. Where the stack exists, this is the port's real
acceptance bar; the wiring tests in `test_qwen2.py` cannot catch a transposed weight
or a wrong rope theta, and this does.

Tolerances: the port computes in f32 from f32-upcast weights; transformers on CPU does
the same when told `torch_dtype=float32`. Differences are reassociation, so the bar is
tight: max |Δlogit| under 2e-2 on a 20-logit spread, NLL to 1e-3.
"""

import numpy as np
import pytest

pytest.importorskip("safetensors")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from shardes.problems import qwen2  # noqa: E402

#: (repo, config constructor). The 1.5B row skips unless fetch.py has cached it;
#: same acceptance bar for every port the experiments rely on.
CHECKPOINTS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen25_05b"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen25_15b"),
]

PROMPTS = [
    "The capital of France is",
    "Using the numbers 3, 5 and 7, make 21:",
]


def _checkpoint_dir(repo):
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        # local_files_only: the suite makes no network calls. fetch.py populates this,
        # and the allow_patterns must match its filter or this demands files that were
        # deliberately never fetched and skips despite a perfectly good cache.
        return snapshot_download(
            repo, local_files_only=True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
        )
    except Exception:
        pytest.skip(f"{repo} not in the local HF cache; run experiments/countdown/fetch.py")


@pytest.fixture(scope="module", params=CHECKPOINTS, ids=lambda c: c[0].split("/")[-1])
def stack(request):
    repo, ctor = request.param
    ckpt = _checkpoint_dir(repo)
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)
    ref = transformers.AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
    ref.eval()
    cfg = getattr(qwen2.Config, ctor)()
    params = qwen2.load(ckpt, cfg, dtype=jnp.float32)
    return tok, ref, cfg, params


def test_the_config_matches_the_checkpoint(stack):
    """Each constructor is a transcription of its config.json; verify, don't trust."""
    _, ref, cfg, _ = stack
    hf = ref.config
    assert (hf.vocab_size, hf.hidden_size, hf.num_hidden_layers) == (
        cfg.vocab, cfg.d_model, cfg.n_layers)
    assert (hf.num_attention_heads, hf.num_key_value_heads) == (cfg.n_heads, cfg.n_kv_heads)
    assert hf.intermediate_size == cfg.d_ff
    # transformers 5.x moved rope_theta into rope_parameters; the config.json field is
    # unchanged. Read either, verify the same number.
    theta = getattr(hf, "rope_theta", None) or hf.rope_parameters["rope_theta"]
    assert theta == cfg.rope_theta and hf.rms_norm_eps == cfg.rms_eps
    assert hf.tie_word_embeddings, "the port assumes a tied head"


def test_logits_match_transformers(stack):
    tok, ref, cfg, params = stack
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="np")["input_ids"].astype(np.int32)
        with torch.no_grad():
            want = ref(torch.from_numpy(ids).long()).logits.float().numpy()
        got = np.asarray(qwen2.forward(params, jnp.asarray(ids), cfg))
        worst = float(np.max(np.abs(got - want)))
        spread = float(want.max() - want.min())
        assert worst < 2e-2, (
            f"{prompt!r}: max |dlogit| {worst:.3e} on a spread of {spread:.1f}; "
            "same weights must give the same logits"
        )


def test_next_token_agrees_everywhere(stack):
    """Greedy argmax at every position: the decision the decode loop will actually take."""
    tok, ref, cfg, params = stack
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="np")["input_ids"].astype(np.int32)
        with torch.no_grad():
            want = ref(torch.from_numpy(ids).long()).logits.argmax(-1).numpy()
        got = np.asarray(qwen2.forward(params, jnp.asarray(ids), cfg).argmax(-1))
        assert (got == want).all()


def test_nll_matches(stack):
    tok, ref, cfg, params = stack
    ids = tok(PROMPTS[0], return_tensors="np")["input_ids"].astype(np.int32)
    t = torch.from_numpy(ids).long()
    with torch.no_grad():
        logits = ref(t).logits[:, :-1].float()
        want = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1)
        ).item()
    got = float(qwen2.nll(params, (jnp.asarray(ids), jnp.ones_like(jnp.asarray(ids))), cfg))
    assert abs(got - want) < 1e-3, f"NLL {got:.5f} vs transformers {want:.5f}"
