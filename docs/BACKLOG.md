# Backlog

Things deferred with a reason, not forgotten. Each entry says what would settle it and what
it blocks, so picking one up does not start with re-deriving why it is here.

Not a wishlist. If an item stops being worth doing, delete it and say so in the commit.

---

## B1 — Why scrambled Sobol degrades with N

**Status**: open. Measured in E1, cause unidentified.
**Blocks**: any public claim about QMC for ES, in either direction.

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

## B3 — Does coupling help on a multimodal objective?

**Status**: open, and G0 does **not** answer it. **Blocks**: nothing; a follow-up.

E1 measured that coupling does not improve the *estimator* on one transformer block. It did
not measure task performance, and `docs/04` C3.3's own caveat cuts both ways: parameter-space
noise acts as Gaussian smoothing, so a better-conditioned estimate can be a worse smoother,
and the classical QMC-for-ES results are strongest exactly where this experiment is silent —
multimodal control tasks.

**What would settle it**: end-to-end runs at matched compute on a multimodal control task,
≥3 seeds, coupled vs uncoupled, reporting variance across seeds.

Keep this distinct from B1. B1 asks whether the Sobol implementation is sound; B3 asks
whether the whole idea has a regime where it pays. A negative on B1 does not resolve B3.

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
