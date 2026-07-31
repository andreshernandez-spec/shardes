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
   would have to be threaded through the `Mirrored` wrapper.

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

---

**Revisited in Phase 1, 2026-07-31.** The constraint held up; what did not was everything
around it. Measuring the failure modes first turned up three problems of very different
severity, and the worst had nothing to do with generality.

**A silent-wrong path, which mattered more than the constraint.** `LowRankWeight` was a
`NamedTuple`, so it inherited tuple semantics: `table[0]` returned the base matrix `w` rather
than row 0, `len(table)` was the field count, and iterating yielded fields. No error, wrong
answer. It is now a registered dataclass — not a sequence — so all three are refused. Same
four pytree leaves, same behaviour under jit/vmap/shard_map.

**Unreadable errors.** All six ways of misusing a weight leaked the NamedTuple: `jnp.einsum`
gave *"setting an array element with a sequence… inhomogeneous"*. The dunders now raise a
message naming `dense` or `embed`.

**The failure arrived too late.** `shardes.check.check_model(model, params, batch)` reports
every leaf the model does not reach through a seam, in a second, on CPU, before a sweep is
booked. It reports *unread* leaves too — a parameter no forward pass touches is still
perturbed and still contracted, so it spends population variance on something that cannot
affect fitness, which is a quieter bug than an unrouted matmul.

**A jaxpr rewriter is still deferred, and now has a trigger rather than a date**: a real user
with a real Flax model. Two alternatives were considered and rejected on evidence rather than
taste, and are recorded so they are not revisited casually:

- *Operator overloading* (`__matmul__`, `.T`, a working `__getitem__`) so natural code works.
  Checked against the three real violations committed while writing Phase 1 — `jnp.einsum`,
  elementwise arithmetic, and `x @ W.T` — and it would have caught **none** of them: einsum
  does not route through `__matmul__`. It adds API surface and a second way to do everything
  while covering the cases that do not occur.
- *A `__jax_array__` fallback* that materializes when interception fails. This converts a loud
  error into a silent `(n, V, d)` allocation. For a library whose claim is not materializing,
  that is the one failure mode worse than an error.

**The constraint is real and it bites.** It was violated three separate times in one session,
each time by writing the obvious thing. It fails loudly every time, which is why it is a cost
rather than a defect — but it is the first thing a porting user meets, and `check_model` exists
because "safe" and "discoverable" are different properties.

Implementations:

| Name | Rank | Materializes? | Corresponds to |
|---|---|---|---|
| `IIDGaussian` | full | yes | textbook OpenAI-ES |
| `SeedRegenerated` | full | transiently, per layer | Qiu et al. |
| `LowRank(r)` | `r` | **no** | EGGROLL |

Plus a wrapper, not a fourth implementation:

```python
Mirrored(inner)      # antithetic pairs; halves effective N
```

`Mirrored` is not optional — it's the honest baseline, since mirrored sampling is standard
in ES. Reporting a win against unmirrored i.i.d. is reporting a known result.

**Sample design across members is a constructor argument, not a second wrapper.** This was
originally scoped as `Coupled(inner, kind)`; it cannot be built that way, because coupling
changes the perturbation *directions* while `Mirrored` only changes their signs, and a
wrapper would have to reach into an opaque inner perturbation and replace the noise it just
drew. So it is a `shardes.coupling.Coupling` handed to the strategy:

```python
IIDGaussian(coupling=OrthogonalHD())
Mirrored(LowRank(r=1, coupling=ScrambledSobol()))
```

`Gaussian()` is the uncoupled default and is bitwise what the strategies did before coupling
existed. The reasoning is in `src/shardes/coupling.py`; `docs/04-phase3-coupling.md` C3.1
records it as the finding that capability asked for.

`ScrambledSobol` is **low-rank only**. See the sweep grid in C0.5 for why.

### C0.2 — Contraction cost is understood, not guessed

For `LowRank`, `contract` is `Σₙ wₙ aₙ bₙᵀ = (A ⊙ w) Bᵀ`, one `(m × Nr) × (Nr × n)` GEMM
— `m·n·N·r` FLOPs. At `m = n = 4096`, `N = 2¹⁸`, `r = 1`: 4.4 TFLOP, ≈6 ms on an H100.
Fine. But storing `A` and `B` costs `N·r·(m+n)` ≈ 2 GB/layer in bf16. Note in the results
whether you hit that; it motivates the seed-regeneration-inside-low-rank synthesis
(`docs/00-context.md`).

**Measured, and it is not `A` and `B` that bind.** On the RTX 3080 (16 GB, ~12.2 GB usable
after JAX's preallocation) at `m = n = 512`, unchunked:

| strategy | first `N` that OOMs |
|---|---|
| `iid_gaussian` | 1,024 |
| `lowrank_r1` | 4,096 |
| `seed_regenerated` | none up to 2¹⁸ |

`lowrank_r1` OOMing at all is the point. The *perturbation* stays factored exactly as
designed: `A` and `B` at `N = 4096, r = 1` are 100 MB together, measured from their shapes.
What blows up is the **forward pass**. `apply` vmaps the model over members, so the block's
activations carry a members axis: `batch·seq = 256` tokens × 512 × 4 B is 512 KB per
intermediate per member, and the block has six projections plus attention scores, so roughly
4–5 MB live per member (derived, not measured). At `N = 4096` that is 16–20 GB against a
100 MB perturbation — the activations bind **~86× harder**, and invariant 3 does nothing about
them because it is a statement about the perturbation.

So `chunk` in `experiments/phase0/config.yaml` is not a tuning knob, it is what makes the
sweep runnable at all. It bounds the member axis in both of `estimate`'s passes. With
`chunk = 256` every strategy runs to `N = 2¹⁸` with no OOM.

`seed_regenerated` never OOMs because it already scans one member at a time. That is Qiu's
bet paying off on the axis nobody advertises: it buys activation memory, not just
perturbation storage.

### C0.3 — An FWHT with an exact correctness oracle

Needed for `OrthogonalHD`: the scalable orthogonal construction is `HD₁HD₂D₃…`, products of
Hadamard transforms and Rademacher diagonals. A reference `O(n log n)` butterfly is enough
here; a Mosaic GPU kernel is out of scope for Phase 0.

**The oracle ships in JAX**: `jax.scipy.linalg.hadamard(n) @ x`. It is a dense `O(n²)`
Sylvester constructor, so it's useless as an implementation and perfect as a test.

The coupling only ever needs *one row* of the product, so it never forms the matrix: row `p`
of `H` is `(-1)^popcount(p & j)` in `O(d)`, then `factors - 1` butterflies and `factors` sign
flips. `shardes.coupling.hadamard_row` is that row, checked against `fwht` of a one-hot and
so transitively against the dense oracle.

### C0.4 — Test models with analytic or backprop gradients

Three, in increasing realism:

1. **Quadratic** `f(θ) = ½ θᵀHθ` with known `H`. Gradient is analytic, no autodiff
   involved. Catches sign errors, scaling errors, and unbiasedness bugs. Should be the
   first thing that passes.
2. **Small MLP** on synthetic regression. Backprop oracle. Introduces pytree structure.
3. **One transformer block**, `d_model` set so `d_eff = m + n` is realistic. This is the
   configuration the result is actually about.

For (3), pick shapes so you can sweep `N/d_eff` across three orders of magnitude on one
GPU. `m = n = 512` with six square matrices (q, k, v, o, up, down) and no learnable norms:

| panel | `d_eff` | `N/d_eff` over `N ∈ {2⁶ … 2¹⁸}` |
|---|---|---|
| full rank | `6·512·512` = 1,572,864 | 4e-5 → **0.167** |
| rank 1 | `6·(512+512)` = 6,144 | 0.010 → **42.7** |

`d_eff` **sums over the whole params tree**, because one member's `ε` covers every matrix
at once. Full rank stays well under `N/d_eff = 1` and rank 1 crosses it by ~40× either
way, which is what G0 needs. Regenerate with `tests/test_dimensions.py`.

The two panels' `d_eff` measure different quantities, deliberately: see
`src/shardes/dimensions.py`. Under full rank the space you sample in *is* the space `∇f`
lives in; under rank 1 you draw 6,144 numbers whose products still live in `ℝ^1,572,864`.
`d_eff` is the dimension **sample design operates in**, which is the quantity coupling's
leverage scales with. Put that in the F5 caption, or a reader will think the panels aren't
comparable.

`d_ff = d_model` is not a realistic transformer ratio (usually 4×). It's chosen so every
matrix has the same `d_eff` and the x-axis is one number rather than a mixture. Say so in
the limitations.

At `N = 2¹⁸` storing `A` and `B` is `262144 × 512 × 4 B ≈ 0.5 GB` each in f32, which is
comfortable. Do **not** jump straight to `m = n = 4096`; you'll be memory-bound before
you're in the interesting regime.

### C0.5 — Metrics and the sweep driver

Per configuration, over `R ≥ 30` independent replicates:

- **Cosine similarity** `cos(ĝ, ∇f)` — mean and IQR. The headline metric; it's what
  actually determines whether the update direction is useful.
- **Relative MSE** `E‖ĝ − ∇f‖² / ‖∇f‖²`.
- **Bias check** `‖E[ĝ] − ∇f‖ / ‖∇f‖` over replicates. Must go to zero for every
  *sampling* scheme. If it doesn't for scrambled Sobol, the scrambling is wrong.

  **It is a correctness gate only on the `shaping ∈ {none, centered}` slice.** Centered
  ranks are not estimating `∇f` at all; they're a deliberately different update direction,
  so their bias stays large by design and reading it as a failure would be a mistake.
  `test_centered_ranks_is_not_an_unbiased_estimator` asserts it stays large, so the
  distinction can't quietly erode into a bug hunt.

  Related trap, measured: subtracting the mean fitness without the `n/(n-1)` correction
  estimates `(1 − 1/n)·∇f`, because `f̄` contains `fᵢ` and correlates with `εᵢ`. At
  `n = 30` that's a 3.3% systematic underestimate that reads as a slightly wrong learning
  rate. `shardes.shaping.centered` carries the correction; the naive version is pinned as
  biased in `test_naive_mean_subtraction_would_be_biased`.

  `centered` (mean-subtracted, corrected, unbiased) is implemented but **not** in the sweep
  axis below. Adding it would separate "variance reduction" from "rank transform", which
  are currently confounded in the `none → centered_ranks` jump. Worth the extra 1.5× of
  configs, but that's a compute-budget call.
- Wall-clock per estimate, for context. Not the point of this phase.

Sweep axes: `N` × `rank ∈ {full, 4, 1}` × `scheme` × `shaping` × `σ ∈ {3 values}`.

**Three of those axes are conditional rather than crossed.** `scheme` on `rank`, `shaping` on
`scheme`, and `population` on `rank`. All three are encoded once, in
`src/shardes/strategies/registry.py` and `experiments/phase0/config.yaml`, and
`tests/test_phase0_driver.py` pins each one in both directions.

| rank | populations | why it stops there |
|---|---|---|
| full | 2⁶ … 2¹⁴ | `N/d_eff` tops out at 0.010, two decades below the line it exists to sit under. Full-rank `orthogonal_hd` costs >400 s per replicate at 2¹⁸, so that cell cannot reach `R = 30` on one GPU: the 900 s cap would record 2 replicates. Measured, `docs/04` C3.3. |
| 4, 1 | 2⁶ … 2¹⁸ | Crosses `N/d_eff = 1` near `N = 6144` and reaches 42.7. This is the regime the figure is about, and it is cheap: 10.8 s per replicate at 2¹⁸. |

**The scheme grid is not rectangular either:**

| rank | schemes |
|---|---|
| full | `iid`, `mirrored`, `mirrored+orthogonal_hd` |
| 4, 1 | `iid`, `mirrored`, `mirrored+orthogonal_hd`, `mirrored+sobol` |

**`shaping = none` is a dead arm on the transformer block and should be replaced by
`centered`.** Measured at `d_model=64`, `N=128`, `σ=0.01`, `R=200`: raw fitness gives
relative bias 255 and `cos(E[ĝ], ∇f) = 0.008`; mean-subtracted-and-corrected gives bias
0.994 and cosine 0.708. The estimator is unbiased either way, but `f(θ) = 2.65` against
`‖∇f‖ = 1.02` and raw fitness divides that constant by `σ`, so the variance is ~250×
larger and R would have to be around `10⁵` to see anything. The sweep budgets 30.

`centered` also separates variance reduction from the rank transform, which `none →
centered_ranks` confounds. Recommended axis: `shaping ∈ {centered, centered_ranks}`, with
`none` kept only on the quadratic where it is cheap and exactly unbiased.

**But `centered` must not be paired with a mirrored scheme.** The antithetic pair already
cancels `f̄`, so the `n/(n-1)` factor over-corrects and the estimator targets
`n/(n-1)·∇f`: measured 6.7% the wrong way at `n = 16`, worse as `n` shrinks. The
correction belongs to the estimator-and-shaping *pair*, not to shaping alone. So the
shaping axis is itself conditional on the scheme:

| scheme | shaping |
|---|---|
| `iid` | `centered`, `centered_ranks` |
| `mirrored`, `mirrored+*` | `none`, `centered_ranks` |

`none` under mirroring is already centred by construction, so it is the right unshaped
baseline there rather than the noisy one it is on `iid`. Asserted in both directions in
`test_naive_mean_subtraction_would_be_biased`.

`mirrored+sobol` is absent from the full-rank row because it cannot be built. Full-rank
sampling is in `ℝ^{mn}`, and every published direction-number table stops around 20k
dimensions (cited: cuRAND documents 20,000; `scipy.stats.qmc.Sobol.MAXDIM` is 21,201).
Extending Joe-Kuo past that is its own research problem, not a Phase 0 task.

This is a second, independent reason coupling only has room to work under low rank. The
first is the `N/d_eff` argument in `docs/00-context.md`: coupling has no *leverage* at
`N ≪ d`. This one is sharper: Sobol is not *constructible* at `d = mn` at all.

**Two reasons to expect the sobol arm to underperform, on the record before the run.** Both
are properties of the method rather than of the implementation, so a null result there is a
prediction and not an excuse:

- Sobol's guarantee rests on **low effective dimension**. `f(θ + σε)` depends on every
  coordinate of `ε` roughly equally, which is the worst case for QMC and the case ES is in by
  construction.
- Sobol's **2-D projections degrade in the later dimensions**. At `m = 512` the design is a
  512-dimensional point set, well inside the range where that is a documented weakness of
  Joe-Kuo tables rather than a subtlety.

`orthogonal_hd` has no analogue of either, which is a third reason it is the curve that
carries the G0 comparison across both panels.

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

**One figure.** Log–log, x-axis `N/d_eff`, y-axis `cos(ĝ, ∇f)`. Two panels: full-rank
and rank-1. Three curves in the full-rank panel, four in the rank-1 panel, one per scheme,
IQR bands over replicates. A vertical line at `N/d_eff = 1`.

**The y-axis was `1 − cos` and is now `cos`, because of what the measurement turned out to
be.** `1 − cos` is the right transform when cosine approaches 1: it turns "almost perfect"
into a readable decade. On this block cosine spans about **0.008 to 0.1**, so `1 − cos` lands
in `[0.9, 1.0]` and a log axis spends its entire range on the third decimal. Plotting cosine
directly gives over a decade of legible range on the same data. `plot.py --y one-minus-cos`
still produces the original, for a model that ever gets close enough to 1 for it to mean
something.

**Measured, and worth stating because it validates the estimator independently of G0:** in
the full-rank panel `cos ≈ √(N/d_eff)` to within the marker size — 0.1 at `N/d_eff = 10⁻²`,
0.0063 at `4·10⁻⁵`, slope ½ on log–log. That is the textbook ES scaling, and getting it for
free is a stronger check on the harness than any single unit test.

**The two panels' x-axes are different quantities and the caption must say so** (see
`src/shardes/dimensions.py`). Reading across them at equal `N` rather than equal `N/d_eff`:
at `N = 16384` full rank reaches `cos ≈ 0.1` and rank 1 reaches `≈ 0.045`, so the factored
perturbation costs roughly 2.2× in estimator quality per member. That is the EGGROLL
tradeoff, priced.

**F5 selects the shaping arm by role, not by name.** The conditional shaping axis leaves no
single mode common to all four schemes, so `plot.py` defaults to `--shaping baseline`, which
means `centered` on the iid side and `none` under mirroring: each scheme's unbiased,
variance-reduced arm. That is the slice where `ĝ` is actually estimating `∇f`, and therefore
the only one where coupling has any right to show an effect. `--shaping centered_ranks` is the
supporting comparison, kept separate because `docs/00` obstacle 2 predicts rank shaping erodes
QMC's advantage and leading with it would bias the figure toward a null.

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

## The answer: **no**

> *Drafted by Claude Code on 2026-07-30 at Andres's request, because he was away from the
> machine. Ground rule 1 says this paragraph is his. The numbers and the reasoning are
> checked and reproducible; the wording is a placeholder to be rewritten in his voice.*

**Rank-1 curves do not separate across sampling schemes, at any `N/d_eff` this hardware
reaches.** `orthogonal_hd` lies on top of the uncoupled baseline everywhere: cosine ratios
0.99–1.01 with overlapping IQRs at every population, sigma and rank, out to `N/d_eff = 42.7`.
The full-rank panel does not separate either, so the control behaves as predicted; the
difference the gate was looking for is absent on both sides.

Evidence: `experiments/phase0/`, 456 configs, `R = 30` each, one uniform environment (commit
`1fb6743`, RTX 3080, jaxlib 0.11.0), 13.07 h. Figure `figures/f5-estimator-quality.png`,
comparison `gate.py`.

**The null is not a failure to apply the treatment, and that was checked rather than
assumed.** A 512-member `orthogonal_hd` block is an *exactly* orthonormal basis of `ℝ⁵¹²`
(max off-diagonal Gram entry `0.0e+00`) where the i.i.d. block reaches `0.215`, and the two
strategies' contractions differ by 1.4 relative. The designs are as different as two designs
can be, and the estimator cannot tell them apart.

**Why, in one line.** Measured cosine tracks `√(N/d_ambient)`: at full rank, `N = 16384`,
predicted `0.1021` against measured `0.1013`, and every curve in F5 is slope ½ on log–log. If
estimator quality is set by how many members you have and how large the model is, there is
nothing left for *how you choose them* to influence. The null and the power law are one fact.

**Where the original reasoning went wrong.** The `N/d_eff` argument established that low rank
creates *room* for sample design, and it does — rank 1 genuinely reaches `N/d_eff = 42.7`.
That was taken as evidence a *mechanism* would appear. It is not the same claim. The tell was
available before any GPU time: coupling leaves `E[εεᵀ]` and the pairwise cross-moments
unchanged (`tests/test_coupling.py::test_hd_is_uncorrelated_across_members_within_a_block`),
so it cannot move the variance of a linear functional. Only higher-order structure was ever in
play, and on this objective it pays nothing.

### Consequences

- The strategy abstraction **stays**, justified by there being two real algorithms and two
  ranks, not by coupling. It stays *thin*: coupling is a constructor argument and the sharded
  core never learns about it.
- **Phase 3 is dropped.** See `docs/04-phase3-coupling.md`.
- `OrthogonalHD` and `ScrambledSobol` stay in the library. They cost nothing to keep, the
  property suite covers them, and `docs/BACKLOG.md` B3 is the question they would be needed
  for.

### What this result does *not* say

- It does not say coupling cannot help **an optimizer**. E1 measured estimator quality, and
  `docs/04` C3.3's caveat cuts both ways: parameter-space noise does optimization work as
  Gaussian smoothing, so a better-conditioned estimate can be a worse smoother. Task-level
  validation on a multimodal objective is `docs/BACKLOG.md` B3 and is open.
- It does not say scrambled Sobol hurts. Sobol *was* the one scheme that separated, and the
  wrong way (0.892 at `N = 2¹⁸`, rank 1) — and **the cause was found and fixed on 2026-07-31**:
  a digital shift alone does not decorrelate streams, so every stream reused one inter-member
  arrangement and its deficiency added coherently. `ScrambledSobol` now draws a different block
  of direction numbers per stream, which recovers the loss (`docs/BACKLOG.md` B1, closed).

  **E1's sobol arm therefore measures a construction that no longer ships.** Its numbers stand
  as a measurement of `blocks=1` and are correct as such; they are not a property of scrambled
  Sobol and must not be quoted as one. The arm has not been re-run: it would cost ~2 h to
  change one curve in a figure whose gate already came back negative for a different scheme.
- It is one transformer block, not an LLM. The `N/d_eff` regime transfers; the loss landscape
  does not.

### Findings worth keeping that G0 did not ask for

- **Mirroring is not a free win and its sign depends on sigma.** At rank 1 it is 1.67×
  *better* than i.i.d. at `σ = 1e-2` and ~1.4× *worse* at `σ = 1e-3`. It cancels the even part
  of `f`, which matters only once sigma is large enough to sample curvature; below that,
  centering already removes the constant and mirroring merely spends the population on half as
  many distinct directions.
- **The rank-1 restriction is priced.** At equal `N`, full rank reaches `cos ≈ 0.10` where
  rank 1 reaches `0.045`: about 2.2× in estimator quality per member.
- **Coupling's cost is two numbers**, +4.2% at rank 1 and +770% at full rank, because the
  design dimension differs by 512×. `docs/04` C3.3.
- **`σ = 0.1` is a dead arm on this block** (cos ~1e-3, occasionally negative), the same shape
  of finding as `shaping = none`. Drop it from any successor sweep.

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
