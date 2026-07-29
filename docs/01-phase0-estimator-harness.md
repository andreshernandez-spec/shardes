# Phase 0 — Estimator harness

**Compute**: 1 GPU, ~1 day of actual runtime. Develop on CPU.
**Duration**: 1–2 weeks.
**Gate**: G0 — see the bottom of this file.

---

## Goal

Measure the statistical efficiency of ES gradient estimators against an **exact oracle**,
as a function of population size, perturbation rank, and sampling scheme.

The oracle is the point. Use a differentiable model, so backprop gives you the true `∇f`
directly — no proxy metric, no reference-estimator-with-huge-N, no ambiguity about what
"good" means.

## Why this is first

It settles the library's central abstraction before three months of work depend on it:

- If coupling matters under low-rank perturbation, "perturbation strategy" is a
  first-class, pluggable, shape-aware component and the API is designed around it.
- If it doesn't, the abstraction is still needed (two algorithms, two schemes) but stays
  thin, and Phase 3 is dropped.

It is also a vertical slice of the library, not a detour: it requires a batched low-rank
forward pass, a fitness-evaluation loop, and the strategy interface. All three are kept.

---

## Capabilities delivered

By the end of Phase 0, the repo can do all of the following from a clean checkout.

### C0.1 — The strategy protocol exists and has three implementations

The interface that everything else hangs off. Sketch — refine it, don't copy it:

```python
class PerturbationStrategy(Protocol):
    def sample(self, key: Array, params: PyTree, n: int) -> Perturbation:
        """Opaque per-member perturbation state for `n` members.
        Shape-aware: leaves keep their (m, n) structure. No ravel_pytree."""

    def apply(self, params: PyTree, pert: Perturbation, sigma: float) -> Callable:
        """Return a callable evaluating the model for all n members.
        Full-rank: materialize per member, or regenerate from seed.
        Low-rank:  rewrite x @ W.T  ->  x @ W.T + (x @ B) @ A.T. Never materialize."""

    def contract(self, pert: Perturbation, weights: Array) -> PyTree:
        """Contract shaped fitness weights (n,) into a params-shaped update.
        The only place a full (m, n) tensor is instantiated."""
```

**Four things the sketch doesn't answer.** Each has to be settled before the protocol is
written, and each has more than one defensible answer, so they're flagged rather than
picked. Listed worst-first.

1. **`sample(key, params, n)` cannot express the seed contract under sharding.** Member `i`
   derives from `fold_in(base_key, i)` with `i` the *global* index. Called inside
   `shard_map`, each device sees only its local `n`, so every device would generate members
   `0…n/D-1` and produce identical perturbations. Device-count invariance is invariant 2 in
   `CLAUDE.md` and this signature can't satisfy it. Either `sample` takes the member
   indices (or a global offset) explicitly, or it is only ever called outside the shard and
   the result is sharded afterwards, which costs the low-rank path its whole point. This is
   the one to decide first.

2. **`apply` never receives the model.** It returns "a callable evaluating the model for
   all `n` members" given only `params`, `pert` and `sigma`. For `LowRank` the entire trick
   is rewriting `x @ W.T` into `x @ W.T + (x @ B) @ A.T` *inside* the forward pass, so the
   strategy has to reach the model's matmuls. Three routes: the model is written against a
   strategy-supplied linear op; a jaxpr interpreter rewrites `dot_general`; or the strategy
   walks a known module structure. The choice decides how invasive the library is on user
   code, which is most of what adoption turns on.

3. **Where does `sigma` live?** It's an argument to `apply` but not `sample`, so
   perturbations are presumably unit-scale and scaled at application. Then `contract` needs
   it too, because the estimator is `(1/(Nσ)) Σ wₙ εₙ`. Either `sigma` is threaded through
   all three or it belongs to the perturbation. Getting this inconsistent is a silent
   scale error in the gradient, not a crash.

4. **What does `Perturbation` have to carry?** Contraction Strategy A regenerates all `N`
   perturbations from seeds on every device, so `contract` needs enough to rebuild them
   (seeds plus the params shapes), while Strategy B only needs the local shard's factors.
   If one type serves both, it carries the union. If not, `contract` is per-strategy-pair
   and the matrix of implementations doubles.

Implementations:

| Name | Rank | Materializes? | Corresponds to |
|---|---|---|---|
| `IIDGaussian` | full | yes | textbook OpenAI-ES |
| `SeedRegenerated` | full | transiently, per layer | Qiu et al. |
| `LowRank(r)` | `r` | **no** | EGGROLL |

Plus a wrapper, not a fourth implementation:

```python
Mirrored(inner)      # antithetic pairs; halves effective N
Coupled(inner, kind) # kind in {"none", "orthogonal_hd", "sobol_scrambled"}
```

`Mirrored` is not optional — it's the honest baseline, since mirrored sampling is standard
in ES. Reporting a win against unmirrored i.i.d. is reporting a known result.

`kind="sobol_scrambled"` is **low-rank only**. See the sweep grid in C0.5 for why.

### C0.2 — Contraction cost is understood, not guessed

For `LowRank`, `contract` is `Σₙ wₙ aₙ bₙᵀ = (A ⊙ w) Bᵀ`, one `(m × Nr) × (Nr × n)` GEMM
— `m·n·N·r` FLOPs. At `m = n = 4096`, `N = 2¹⁸`, `r = 1`: 4.4 TFLOP, ≈6 ms on an H100.
Fine. But storing `A` and `B` costs `N·r·(m+n)` ≈ 2 GB/layer in bf16. Note in the results
whether you hit that; it motivates the seed-regeneration-inside-low-rank synthesis
(`docs/00-context.md`).

### C0.3 — An FWHT with an exact correctness oracle

Needed for `Coupled(kind="orthogonal_hd")`: the scalable orthogonal construction is
`HD₁HD₂D₃…`, products of Hadamard transforms and Rademacher diagonals. A reference
`O(n log n)` butterfly is enough here; a Mosaic GPU kernel is out of scope for Phase 0.

**The oracle ships in JAX**: `jax.scipy.linalg.hadamard(n) @ x`. It is a dense `O(n²)`
Sylvester constructor, so it's useless as an implementation and perfect as a test.

### C0.4 — Test models with analytic or backprop gradients

Three, in increasing realism:

1. **Quadratic** `f(θ) = ½ θᵀHθ` with known `H`. Gradient is analytic, no autodiff
   involved. Catches sign errors, scaling errors, and unbiasedness bugs. Should be the
   first thing that passes.
2. **Small MLP** on synthetic regression. Backprop oracle. Introduces pytree structure.
3. **One transformer block**, `d_model` set so `d_eff = m + n` is realistic. This is the
   configuration the result is actually about.

For (3), pick shapes so you can sweep `N/d_eff` across three orders of magnitude on one
GPU. Suggested: `m = n = 512` → `d_eff = 1024`, sweep `N ∈ {2⁶ … 2¹⁸}` → `N/d_eff ∈
[0.06, 256]`. At `N = 2¹⁸` that's `262144 × 512 × 4 B ≈ 0.5 GB` each for `A` and `B` in
f32 — comfortable. Do **not** jump straight to `m = n = 4096`; you'll be memory-bound
before you're in the interesting regime.

### C0.5 — Metrics and the sweep driver

Per configuration, over `R ≥ 30` independent replicates:

- **Cosine similarity** `cos(ĝ, ∇f)` — mean and IQR. The headline metric; it's what
  actually determines whether the update direction is useful.
- **Relative MSE** `E‖ĝ − ∇f‖² / ‖∇f‖²`.
- **Bias check** `‖E[ĝ] − ∇f‖ / ‖∇f‖` over replicates. Must go to zero for every scheme.
  If it doesn't for scrambled Sobol, the scrambling is wrong.
- Wall-clock per estimate, for context. Not the point of this phase.

Sweep axes: `N` × `rank ∈ {full, 4, 1}` × `scheme` × `shaping ∈ {none, centered_ranks}` ×
`σ ∈ {3 values}`.

**The grid is not rectangular:**

| rank | schemes |
|---|---|
| full | `iid`, `mirrored`, `mirrored+orthogonal_hd` |
| 4, 1 | `iid`, `mirrored`, `mirrored+orthogonal_hd`, `mirrored+sobol` |

`mirrored+sobol` is absent from the full-rank row because it cannot be built. Full-rank
sampling is in `ℝ^{mn}`, and every published direction-number table stops around 20k
dimensions (cited: cuRAND documents 20,000; `scipy.stats.qmc.Sobol.MAXDIM` is 21,201).
Extending Joe-Kuo past that is its own research problem, not a Phase 0 task.

This is a second, independent reason coupling only has room to work under low rank. The
first is the `N/d_eff` argument in `docs/00-context.md`: coupling has no *leverage* at
`N ≪ d`. This one is sharper: Sobol is not *constructible* at `d = mn` at all.

`orthogonal_hd` has no such ceiling. `HD₁HD₂D₃…` is `O(d log d)` and dimension-agnostic, so
it is the one scheme that appears in both rows, and it is what carries the G0 comparison.

---

## How to test it

`tests/` — CPU, under two minutes, no GPU, no network.

| Test | Asserts |
|---|---|
| `test_fwht_matches_dense` | `fwht(x) == hadamard(n) @ x` to f32 tolerance, `n ∈ {2…4096}` |
| `test_fwht_involution` | `fwht(fwht(x)) == n·x` |
| `test_hd_product_orthogonal` | Gram matrix of `HD₁HD₂D₃` columns ≈ `I` within the expected `O(1/√d)` band |
| `test_quadratic_estimator_unbiased` | For each strategy, `mean(ĝ)` over many seeds → `Hθ`, rel. error < 2% at `R = 2000` |
| `test_lowrank_never_materializes` | Trace the jaxpr under `LowRank`; assert no array of shape `(n, m, k)` with `m,k > r` appears |
| `test_lowrank_converges_to_fullrank` | `LowRank(r)` estimator → `IIDGaussian` estimator as `r` grows; gap decreasing, consistent with `O(1/r)` |
| `test_contract_matches_naive` | Vectorized `contract` == explicit Python loop `Σ wₙ aₙ bₙᵀ` |
| `test_mirrored_cancels_odd` | On an odd `f`, mirrored estimator variance ≪ i.i.d. |
| `test_shaping_is_permutation_equivariant` | Centered ranks depend only on ordering |
| `test_sobol_first_two_moments` | Scrambled Sobol mapped through `Φ⁻¹` has correct mean/covariance |

Two properties that are easy to get wrong and worth their own tests:

- **Seed derivation is by member index.** `sample(key, params, n)[i]` must not depend on
  how members are batched. This is dead code in Phase 0 and load-bearing in Phase 1;
  establish it now.
- **Unbiasedness survives coupling.** Deterministic Sobol makes the estimator biased at
  fixed `N`. Scrambling restores unbiasedness. The bias-check metric is what catches this.

---

## How to showcase it

**One figure.** Log–log, x-axis `N/d_eff`, y-axis `1 − cos(ĝ, ∇f)`. Two panels: full-rank
and rank-1. Three curves in the full-rank panel, four in the rank-1 panel, one per scheme,
IQR bands over replicates. A vertical line at `N/d_eff = 1`.

The claim the figure either supports or kills: *curves separate in the rank-1 panel to the
right of the line, and do not separate in the full-rank panel to the left of it.*

`mirrored+orthogonal_hd` is the curve to read across both panels. It is the same scheme at
two ranks, so if it separates on the right and not on the left, the regime is doing the
work rather than the scheme. That is a cleaner controlled comparison than contrasting two
different sets of schemes, which is what the original rectangular grid would have given.

Supporting table: relative MSE at three `N` values, with and without fitness shaping,
because obstacle (2) in `docs/00-context.md` predicts shaping erodes the advantage and
that prediction should be on the record either way.

Deliverables:

- `experiments/phase0/` — the sweep driver, config committed, results as a parquet/CSV.
- `experiments/phase0/README.md` — GPU model, JAX version, commit SHA, wall-clock,
  and the figure.
- A paragraph in the top-level README stating what was found, including if it was nothing.

---

## Exit criteria — Gate G0

**Answer this question in one sentence, with the figure attached:**

> Do rank-1 estimator-quality curves separate across sampling schemes at `N/d_eff ≳ 1`,
> when full-rank curves at `N/d_eff ≪ 1` do not?

- **Yes** → the strategy abstraction is load-bearing. Design the Phase 1 API around it.
  Phase 3 is live.
- **No** → the abstraction is still needed for the two algorithms but stays thin. Drop
  Phase 3, reclaim a month, and say so in the README.
- **Ambiguous** → record it, proceed to Phase 1, revisit after Phase 2 with real tasks.

**All three outcomes are acceptable results and all three get written up.** A clean
negative here is worth more than a hedge: it's a measurement nobody has published, and it
saves a month.

---

## Known limitations to state up front

- Estimator MSE is **not** a proxy for task performance. See the smoothing caveat in
  `docs/00-context.md`. G0 gates an API decision, not an algorithmic claim.
- A single transformer block is not an LLM. The `N/d_eff` regime transfers; the loss
  landscape does not.
- `R = 30` replicates gives wide error bars at large `N`. Report the IQR, don't hide it.
- Full-rank variance reduction beyond mirrored and `orthogonal_hd` is **deliberately out of
  scope**, not overlooked. The ZO literature has many candidates (control variates,
  subspace projection, preconditioning, importance mixing) and none of them is what this
  phase is asking about. Say so in the writeup rather than letting a reader assume the
  full-rank panel is a best effort.
