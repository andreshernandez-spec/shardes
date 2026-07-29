# Phase 0 — Estimator harness

**Compute**: 1 GPU, ~1 day of actual runtime, on the local RTX 3080 (runbook tier T2).
Develop on CPU.
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

The interface that everything else hangs off. **Settled**; the reasoning is below.

```python
class PerturbationStrategy(Protocol):
    def sample(self, base_key: Key, params: PyTree, member_ids: Array) -> Perturbation:
        """Unit-scale perturbation for exactly the members in `member_ids`.

        member_ids: (n_local,) int32 of GLOBAL member indices. Member i depends only on
        (base_key, i, params shapes). Never on n_local, on where i sits in the array,
        on the device, or on any counter. Leaves keep their (m, n) structure."""

    def apply(self, model: Callable, params: PyTree, pert: Perturbation,
              sigma: float) -> Callable:
        """Given the user's model(params, x) -> y, return g(x) -> (n_local, ...)
        evaluating every member in `pert`.

        Full-rank: materialize per member, or regenerate from seed.
        Low-rank:  rewrite x @ W.T -> x @ W.T + (x @ B) @ A.T. Never materialize."""

    def contract(self, pert: Perturbation, weights: Array) -> PyTree:
        """sum_i weights[i] * eps_i, params-shaped, unit scale.

        weights: (n_local,) shaped fitness. Accumulate in f32 even when the perturbation
        is bf16. The only place a full (m, n) tensor is instantiated."""


class Perturbation(Protocol):
    """Whatever `sample` returns, plus enough state to regenerate itself."""
    base_key: Key
    member_ids: Array
```

**How this was settled.** One decision does most of the work: `sample` takes an explicit
array of global member indices rather than a count.

1. **The global member index is passed in, not derived.** Declare `member_ids` as
   `P("pop")` and the sharding hands each device its own slice. No `axis_index`, no offset
   arithmetic, no code path that behaves differently inside a `shard_map` than outside.
   The seed contract stops being a rule to remember and becomes one there is nowhere to
   break, because `sample` never sees a device index. The test is a line:
   `sample(k, p, [7])` against `sample(k, p, arange(100))[7]`.

   Three other things fall out of the same mechanism, which is the real argument for it.
   **Chunking** is splitting `member_ids` and summing partial `contract`s, which is what
   makes full rank at large `N` possible at all: materializing `2^18` members at
   `m = n = 512` would be 275 TB. **Contraction Strategy A** becomes
   `sample(base_key, params, arange(N))` then `contract`. **Strategy B** becomes `contract`
   on the local shard then `psum`. Same two methods, three lines each.

2. **`sigma` lives in the ES state.** Strategies emit and contract unit-scale
   perturbations. `sigma` enters in exactly two core-owned places: `apply`, where the
   forward pass needs it, and `tell`'s `1/(N sigma)` scaling. `sigma` is a property of the
   distribution, not of the perturbation scheme. It is what the CMA family and every
   adaptive-sigma method updates, so filing it under the strategy would be wrong, and it
   would have to be threaded through each `Mirrored` and `Coupled` wrapper.

   It also keeps the sigma sweep honest. Unit-scale perturbations mean the same directions
   are reused across sigma values, so a gap between sigma arms is the sigma and not a
   different draw. There is a test: `contract` output does not depend on sigma.

   The cost, stated rather than buried: `(pert, sigma)` have to travel together, so no
   single object fully specifies a perturbation. Accepted, because the alternative puts
   sigma in two places and a mismatch there is a silent scale error in the gradient rather
   than a crash.

3. **`apply` takes the model.** The old signature asked it to return a callable evaluating
   a model it had never been given.

4. **One `Perturbation` type serves both contraction strategies**, because it carries
   `(base_key, member_ids)` and can regenerate itself. Strategy A re-derives, Strategy B
   keeps what it has. Regenerability is a requirement of the type, not an accident of the
   implementations.

**Deferred on purpose: how `LowRank.apply` reaches the model's matmuls.** Two routes. A
jaxpr interpreter rewriting `dot_general` works on any model but is fragile under `scan`
and `remat` and has to map primitives back to param leaves. A `shardes.nn.dense` that the
model is written against is easy and correct but means users cannot bring an arbitrary
Flax module.

**Arbitrary-model support is deferred.** Implement `LowRank.apply` against the Phase 0
transformer block, which is ours, and revisit in Phase 1. If the jaxpr route proves
tractable there it generalizes later; if it does not, that was learned on a model we
control rather than on someone's checkpoint. G0 needs statistics, not generality.

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

This grid gets encoded once, in `src/shardes/strategies/registry.py`, which both the test
suite and the E1 driver iterate. Rebuilding it inside `experiments/phase0/run.py` would
give it two homes and one of them would go stale.

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
