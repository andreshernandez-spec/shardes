# Backlog

Things deferred with a reason, not forgotten. Each entry says what would settle it and what
it blocks, so picking one up does not start with re-deriving why it is here.

Not a wishlist. An item that stops being worth doing moves to **Closed** at the bottom with
the reasoning, rather than being deleted: the decision is usually more reusable than the
question was.

---

## B1 — Why scrambled Sobol degrades with N

**Status**: open. Measured in E1, cause unidentified.
**Blocks**: describing `ScrambledSobol`'s behaviour, in either direction.

This is **not** a live coupling question — coupling is settled and closed (B3). This is a
correctness question about a component that ships: `ScrambledSobol` is in the library, passes
the property suite, and demonstrably underperforms for a reason nobody has identified. Either
it has a defect worth fixing or it has a documented weakness worth stating. Right now it has
neither, and that is the problem.

`mirrored_sobol_lr1` is systematically worse than uncoupled `mirrored_lr1`, and the gap
**grows monotonically with N**:

| N | N/d_eff | cos ratio vs uncoupled | IQRs |
|---|---|---|---|
| 2¹⁴ | 2.67 | 0.988 | disjoint |
| 2¹⁶ | 10.67 | 0.966 | disjoint |
| 2¹⁸ | 42.67 | **0.892** | disjoint |

Rank 4 shows the same shape. This is the *only* scheme in E1 that separated from its
uncoupled baseline at all, and it separated the wrong way.

**Ruled out already** (scripts in `experiments/phase0/`, numbers reproducible):

- *Lost design diversity.* No. At N ∈ {512, 2048, 8192, 32768} in d = 512 the Sobol draws
  have full numerical rank 512, zero pairs with |cos| > 0.99, and **lower** mean pairwise
  |cos| than i.i.d. (0.0296 vs 0.0352 at N = 512). The design is better spread, as QMC
  should be.
- *Marginal scale or shape error.* No. `E[x²] = 1.00000` against i.i.d.'s 0.99978,
  `mean = -0.00000`, `E[x⁴]/3 = 0.99999`. Marginals are exact, and better than i.i.d.'s.

**Leading hypothesis, untested.** A digital shift *translates* a point set without changing
its geometry: `{xᵢ ⊕ s}` has the same pairwise XOR structure as `{xᵢ}` for any shift `s`. So
every one of the 12 streams (6 leaves × {a, b}) carries the **same inter-member design
pattern**, merely relabelled, where i.i.d. gets an independent configuration per stream. Any
deficiency in that one pattern would then add coherently across the whole params tree instead
of averaging out — and coherent addition across a growing population is the shape of a defect
that worsens with N.

**What would settle it**: run the sobol arm on a **single-leaf** model. If the harm largely
disappears with one stream, the cause is cross-stream coherence and therefore *my
construction*, not QMC. If it persists, it is a property of Sobol in d = 512 at these N and
the a-priori reasons in `docs/01` C0.5 (high effective dimension, degraded 2-D projections
past a few hundred dimensions) are the explanation.

**Fix if the hypothesis holds**: give each stream an independent scramble rather than only an
independent digital shift — a linear matrix scramble, or Owen nesting, or simply a distinct
direction-number offset per stream.

**Until then**: do not write "scrambled Sobol hurts ES" anywhere. The honest statement is that
one implementation of it did, on one objective, for reasons not yet established.

---

## B2 — A real FWHT for JAX

**Status**: deferred, out of scope. **Blocks**: nothing here.

`transforms/fwht.py` is a reference butterfly: 18 stages for d = 2¹⁸, each materialising a
fresh array, so it is memory-bound. Measured in E1 (`docs/04` C3.3): full-rank
`orthogonal_hd` costs **+770%** against uncoupled, versus **+4.2%** at rank 1, entirely
because the full-rank design dimension is the whole 512×512 leaf. The coupling costs more
than the model does.

That is the first concrete number behind the claim that a Mosaic GPU FWHT kernel is its own
project. It has three independent consumers — LLM quantization rotations (QuaRot/QuIP#/
SpinQuant), SRHT for randomized linear algebra, and orthogonal ES — and JAX ships only the
dense `O(n²)` `jax.scipy.linalg.hadamard` constructor.

Not needed by this library any more: G0 came back negative, so nothing here depends on
`orthogonal_hd` being fast.

---

## B4 — Embedding layers under low-rank perturbation

**Status**: open, pre-existing, not required for any gate.

EGGROLL's reference implementation raises `NotImplementedError` for the embedding path.
`LowRank` here perturbs any non-rank-2 leaf densely, which is correct but gives up the memory
win exactly where the parameters are largest. A real contribution if cracked.

---

## B5 — Importance mixing

**Status**: deferred. Zero hits across the JAX ecosystem; cheap to add; orthogonal to
everything above. Classical NES sample reuse. Revisit after G2.

## B6 — Adapt the per-coordinate sigma

**Status**: deferred with a design and a trigger, not open-ended.
**Blocks**: nothing. No gate requires it and no claim in `docs/05-paper.md` rests on it.

`sigma` may be a scalar or a params-shaped pytree (`docs/02` C1.4), so per-coordinate step
sizes are *expressible*. Nothing adapts them: sigma is whatever the caller set at `init`, for
every generation. Hand-setting per-layer sigmas is a real technique and works today; learning
them does not.

**The obstacle, which is a fact about the protocol rather than a missing function.** Every
diagonal adaptation rule needs a **second moment** of the perturbation:

| rule | needs |
|---|---|
| SNES | `∇_σ ∝ Σ uᵢ(εᵢ² − 1)` |
| sep-CMA-ES | global σ from CSA, diagonal `C` from a rank-μ update `Σ wᵢyᵢ²` |

`contract(pert, weights)` computes `Σ wᵢεᵢ` and is **linear in the weights**, so no choice of
weights yields `Σ wᵢεᵢ²`. **CSA is not a way around it**: it adapts a *scalar* step size by
comparing `‖p_σ‖` against `E‖N(0,I)‖`, one norm and one number, and there is no published rule
that produces a per-coordinate sigma from the mean shift alone. Inventing one is not the job.

**The design, if it is ever wanted.** Not a fourth protocol method — a second *reduction over
the same perturbation*, so `contract(pert, weights, *, moment=1)` keeps the protocol at three
methods and adds one keyword. Implementable everywhere without breaking invariant 3:

| strategy | `moment=2` |
|---|---|
| `IIDGaussian` | the same einsum on `eps**2` |
| `SeedRegenerated` | square the regenerated noise inside the existing scan |
| `LowRank` | `ε² = (a²)(b²)ᵀ / r`, so the same einsum on squared factors — **still never materializes** the `(n, m, k)` product |
| `Mirrored` | `weights[0::2] + weights[1::2]` — the **sum**, where `moment=1` uses the difference |

That last row is worth keeping whatever happens to this entry: antithetic pairs *cancel* in
the first moment and *reinforce* in the second, which is a clean demonstration that the second
moment carries genuinely independent information rather than restating the first.

**Why deferred.** PLAN.md ground rule 3 — do not build machinery speculatively. No gate needs
it, and G0's finding was that the strategy abstraction should stay thin, so widening it for an
algorithm nothing requires is the same speculative move in a new place. There is also a
measured reason not to hurry: **sigma cancels out of the mean step**, because the estimator
divides by `n·σ`. A diagonal changes exploration and conditioning, not step size, so the payoff
is ill-conditioning robustness specifically — and no measurement yet says this project's
problems are ill-conditioned.

**Trigger**: a benchmark problem where a single global sigma visibly limits progress. Then this
is about half a day, against a real need rather than a guess.

---

---

# Closed

Decisions, kept because the reasoning is the useful part. Reopening one needs a reason that
did not exist when it was closed — not just renewed interest.

## B3 — Does coupling help an optimizer on a multimodal objective? — **CLOSED 2026-07-30**

**Decision: coupling is settled for this project. Not disproven in general; done with.**

Gate G0 measured that coupled sampling does not improve ES gradient *estimates* on a
transformer block, at any rank, sigma or population out to `N/d_eff = 42.7`, with the
treatment verified maximal. Strictly, that leaves a gap: estimator MSE is not task
performance, parameter-space noise acts as Gaussian smoothing so a better-conditioned
estimate can be a worse smoother, and the classical QMC-for-ES wins are on multimodal control
problems that a single transformer block is not. Settling that gap would need end-to-end runs
at matched compute on ≥3 seeds — roughly Phase 3's C3.3, which is the part that was dropped.

**Why it is closed rather than left open.** The gap is real and the project is not going to
close it. Leaving it as an open item would be an invitation to relitigate a gate that was
answered cleanly, and a project that reconsiders every past conclusion does not advance. The
honest thing is to state the boundary of the claim — which `docs/05-paper.md` does, in
"Limitations to state, not bury" — and move on.

**What would justify reopening**: someone publishing a positive coupled-ES task result in a
regime this project can reach, or the sharded core turning out to need a design axis that
coupling happens to supply. Renewed curiosity is not a reason.

**What survives elsewhere**: `OrthogonalHD` and `ScrambledSobol` stay in the library. They are
covered by the property suite, they cost nothing to keep, and removing them would delete the
evidence that the question was asked properly.
