# Proposal: the bf16 dtype policy

**Status: ACCEPTED, 2026-08-14. Policy A, master weights.** Decided by Andres from the
draft below; the implementation shipped in the same PR, for line-by-line review per
ground rule 1.

**The accepted shape**: `ShardedES(..., compute_dtype=jnp.bfloat16)`. The state carries
params at f32 or wider, and one per-generation view cast to `compute_dtype` is what
`ask` samples noise from, what `apply` hands the model, and what the contraction
regenerates from, so the noise the forward pass adds and the noise `tell` correlates
against are the same bits. Only the SGD step touches the master. With no `compute_dtype`
nothing casts, and `init` refuses sub-f32 params rather than let promotion decide.
Per-leaf dtype trees are deferred until a model needs one.

**Implementation surfaced one trap the draft missed**: sigma. The state carries it f32,
and `p + s * e` with an f32 `s` promotes a bf16 forward straight back to f32, undoing
the compute-dtype cast two lines after it happened. `LowRank` already cast its factors
for exactly this reason; `IIDGaussian`, `SeedRegenerated` and LowRank's densely
perturbed leaf path did not, and all three now cast the scale into the leaf's dtype.
The trace-time dtype assert in `tests/test_bf16_policy.py` is what caught it.

One prediction from the draft held up measurably: device-count invariance under bf16
compute passes at the f32 test's 1e-6, not the 1e-2 `docs/conventions.md` reserves for
bf16 accumulation paths, because under A there are no bf16 accumulation paths. The
noise quantizes identically at every device count and every accumulator is f32.

**Amended 2026-08-14, on review.** The draft's fitness section said f32 fitness was
"recorded here so nobody optimizes it later", and recording was not enough: with the
policy in place, a model whose loss reduces in bf16 handed `tell` a bf16 fitness and
nothing raised. Reproduced at 4/8 distinct values in a population of 8, the Q3 collapse
in miniature, reachable through the public API. `tell` now refuses sub-f32 fitness with
the fix in the error message, `transformer_block.loss` reduces with
`dtype=jnp.float32` explicitly, and the probe's Q1 exercises the post-decision
contract (refusal without `compute_dtype`, f32 master with a bf16 model view) rather
than the pre-decision defect it was written against, which `init` now refuses to run.
Q1's original output survives as the quotation in the draft below.

The draft as decided on follows, unchanged.

---

**Status: DRAFT, decision open. Andres decides.** Two defensible designs, written down
with their tradeoffs per the working style in `CLAUDE.md`. A recommendation is stated at
the end and it is only that.

Evidence: `experiments/bf16/probe.py`, CPU, seconds to run. Output quoted below from
commit `a858998`, jax 0.11.0. Rerun it before trusting this document if the dtype
handling has changed since.

---

## Why decide now

Three forcing functions, in increasing order of urgency.

**The ES-vs-GRPO experiment needs it.** Qwen ships bf16 checkpoints. A pilot can carry
fp32 masters at 0.5B and dodge the question, but the experiment design has to say which
dtype the forward runs in, and "whatever happens" is not a design.

**A multi-generation `lax.scan` cannot carry the state today.** `scan` requires the
carry to keep its type. The state does not keep its type (next section), so an outer
scan over generations, the natural shape for a training-loop benchmark, fails on the
carry with a dtype mismatch.

**The current behavior is an accident, not either policy.** It has to be replaced by
something deliberate whichever way the decision goes.

## What the code does today, measured

`probe.py` Q1: initialize with bf16 params, run one `ask`/`apply`/`tell`:

    iid_gaussian   params: bfloat16 -> float32, fitness: float32
    mirrored_seed  params: bfloat16 -> float32, fitness: float32

`tell` steps `p - (lr/(n*sigma)) * u` where `u` is f32, because `contract` accumulates
in f32 (`docs/conventions.md`, and it must). JAX promotion then makes the result f32.
Nothing casts back. So:

- Generation 0 evaluates a bf16 model. Every later generation evaluates an f32 one.
- Any memory or throughput number for a "bf16 model" measures generation 0 only. The
  M-series numbers are unaffected, since those sweeps run f32 end to end, but the first
  bf16 benchmark would have been quietly wrong.
- The silent promotion is also why the scan carry fails: bf16 in, f32 out.

This is a defect with a decision inside it. The fix is one cast in `tell`, and the
decision is which side of the cast the true parameters live on.

## Policy A: master weights

The state carries params in f32 regardless of input dtype. `apply` casts to the model's
compute dtype (the dtype the user handed `init`) at the evaluation seam, once per
generation. `tell` updates the f32 master. The forward runs in bf16; the accumulation
runs in f32.

This is mixed-precision training as everyone practices it, for the same reason.
`probe.py` Q2 measures it for our update shape: T=200 random signed steps of fixed
relative size into a bf16 weight versus an f32 master:

    step 1.0e-02 of |w|:  survival  86.9%
    step 3.9e-03 of |w|:  survival 118.2%
    step 1.0e-03 of |w|:  survival  34.1%
    step 1.0e-04 of |w|:  survival -231.3%

bf16 keeps 8 significand bits, so its relative ulp is 2^-8 = 3.9e-3. Steps an order
below that mostly round away (34% survives); two orders below, what survives is
rounding noise with the wrong sign (-231%). Fine-tuning steps live exactly there:
an effective step of 1e-4 relative is ordinary, and Qiu et al. run hundreds of
generations of them. Under a bf16 accumulator that training signal is mostly deleted.

Costs:

- **Memory**: 6 bytes/param while evaluating (4 master + 2 cast) against 2. At Qwen
  0.5B that is 3 GB against 1. The model is replicated by design (docs/02 C1.4), so
  this multiplies by one, not by devices or members.
- **The cast is per-generation work**: a `|params|` elementwise op. Noise against a
  single member's forward pass.
- The carry is f32, so the input dtype has to be remembered separately (a field on the
  strategy state or a `compute_dtype` on `ShardedES`). Small, real API surface.

## Policy B: preserve the input dtype

The state carries params in the dtype the user handed `init`. `tell` computes the
update in f32, then casts the stepped params back. No master copy, no extra memory,
the carry keeps its type, and what you initialized with is what every generation runs.

Costs:

- **Small updates are deleted**, per Q2 above. Not attenuated: at fine-tuning step
  sizes the surviving movement is rounding noise. An ES run that "converges" under
  this policy at bf16 has mostly stopped moving.
- **Device-count invariance loosens.** The invariance test holds f32 at `rtol=1e-12`.
  A bf16 carry quantizes every generation, so invariance is only meaningful at
  `rtol=1e-2` (`docs/conventions.md`), and the most important test in the repo gets
  weaker on the path that would actually ship at scale.
- It is the cheaper policy only when memory is the binding constraint on the *master*,
  which for this library it is not: the design already spends a replicated model per
  device, and seed regeneration exists precisely so perturbations do not dominate.

## Fitness is a separate decision, and it is already made

`probe.py` Q3, a real population on the d=64 transformer block:

    spread 2.122e-02; distinct in f32: 256/256, in bf16: 2/256

256 members collapse to 2 distinct fitness values in bf16. `centered_ranks` on that is
arbitrary ordering of ties, which the noise-floor postmortem showed becomes a different
update. Whatever the params policy, **fitness must reach `tell` in f32**. Today it
does: the loss reduces in f32 by promotion, `_midranks` casts to f32 defensively, and
`contract` accumulates in f32. The policy decision does not touch this; it is recorded
here so nobody "optimizes" the fitness path to bf16 later.

## What this does not decide

- **Matmul precision** (`jax.default_matmul_precision`): orthogonal knob, stays as is.
- **Perturbation dtype**: strategies already sample in the leaf's dtype and `LowRank`
  casts its factors to it. Under A the leaf is the bf16 cast, so perturbations stay
  bf16 and the memory story of docs/00 is unchanged.
- **The optimizer**: plain SGD today. An optax hook would inherit whichever carry the
  policy picks, which is one more reason the carry should be the f32 one.

## Recommendation

**A, master weights.** The library's stated purpose is ES at LLM scale, where bf16
checkpoints are the input and hundreds of small steps are the workload. Q2 says policy
B deletes that workload. The price is 2 extra bytes per parameter on a design that
already replicates the model deliberately, and the strongest test in the repo stays
strong. B survives as the trivial case of A: an f32 input has an f32 master and the
cast is a no-op.

If accepted, the work is: the cast in `apply`, the cast policy field, a `tell` that
steps the master, a scan-carry test, and a bf16 device-invariance test at the
conventions tolerance. Small, but it touches `core.py`, so per ground rule 1 the
implementation is Andres's to write or to review line by line.
