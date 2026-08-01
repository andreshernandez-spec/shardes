# Conventions

Short, enforceable rules. If something here turns out to be wrong, change it here rather
than working around it in code.

---

## Numerics

**Test against an exact oracle wherever one exists.** The list for this project:

| Thing | Oracle |
|---|---|
| FWHT | `jax.scipy.linalg.hadamard(n) @ x` — dense `O(n²)`, useless as an implementation, perfect as a test |
| `coupling.hadamard_row(p, n)` | `fwht(one_hot(p))`, and so the dense matrix transitively |
| ES estimator unbiasedness | Analytic `∇f` of a quadratic `½θᵀHθ` |
| Estimator quality generally | Backprop through a differentiable model |
| `LowRank.contract` | Explicit Python loop `Σ wₙ aₙ bₙᵀ` at small `N` |
| Low-rank → full-rank | `IIDGaussian` estimator as `r` grows |
| `OrthogonalHD` within a block | Gram matrix `== I` **exactly** (float tolerance), not within a band — the product of orthogonal matrices is orthogonal |
| Sobol point `i` | `scipy.stats.qmc.Sobol(scramble=False)` at scattered `i`, compared as **integers**, so no tolerance is involved |
| Sobol equidistribution | `N = 2^m` points fall one per `2^m` bin, per coordinate, **exactly** — a `(0,1)`-sequence property that an iid draw fails |

Two of those replaced weaker versions that would have passed on a broken implementation: the
`OrthogonalHD` Gram was going to be asserted "within an `O(1/√d)` band", and Sobol was going
to be checked on its first two moments. Prefer the exact form when the construction admits
one; a statistical check on something deterministic is measuring the wrong thing.

Where no exact oracle exists, compare two *independent* implementations with different
failure modes rather than one implementation against itself at higher precision.

**Accumulation dtype.** Perturbations may be bf16. Reductions over `N` members must
accumulate in **f32**. Measured on random normals at `N = 2¹⁸`, against an exact f64
reference: bf16 accumulation gives `1.2e-3` relative error, f32 gives `5.4e-8`. Four orders
of magnitude. Regenerate with `tests/test_metrics.py::test_bf16_accumulates_in_f32`.

Note *how* it fails, because the obvious demonstration doesn't work: XLA reduces pairwise,
not sequentially, so summing `N` identical values in bf16 is **exact** (every partial is a
power of two). The loss shows up only on varied data. A test built on identical inputs
would pass against a bf16 accumulator and prove nothing.

**Tolerances.** State them, don't discover them. f32 exact-oracle comparisons: `rtol=1e-6`.
Device-invariance in f32: `rtol=1e-12` (it should be near-bitwise). bf16 paths: `rtol=1e-2`
and say so.

---

## Randomness

- Typed keys throughout: `jax.random.key`, not the legacy `PRNGKey` uint32 arrays.
- **Member `i`'s perturbation derives from `fold_in(base_key, i)` with `i` the global
  member index.** Not the device index, not a per-device counter, not sequential
  consumption of a key stream. This is what makes device-count invariance and Qiu-style
  seed regeneration both work, and it is the single easiest invariant to break by accident.
- Every experiment records its seeds. A result you can't re-run isn't a result.

---

## Code

- Type-annotate public functions. `PyTree`, `Array`, `Key` aliases in one place.
- Strategies are `Protocol`s, not ABCs — structural typing keeps user-defined strategies
  first-class without inheritance.
- No `ravel_pytree` under `src/`. There is a test asserting this.
- Pure functions with explicit state. `init` / `ask` / `tell` return new state; nothing
  mutates in place. This is not stylistic — `shard_map` requires it.
- Docstrings state shapes. `A: (n_members, m, r)` is worth more than a paragraph of prose.
- No dependency added without a one-line justification. `jax`, `numpy`, `pytest` are the
  floor. `optax`, `flax`, `chex` acceptable if genuinely used.
- `matplotlib` and `pyyaml` are optional `experiments` extras, not core dependencies. The
  library must install with only `jax` and `numpy`. The test suite covers the experiment
  drivers, so development needs `pip install -e ".[dev,experiments]"`.
- Experiment configs are YAML, and **`yaml.safe_load` is not safe from coercion**. PyYAML
  reads `no`/`No`/`NO`/`off`/`on`/`yes` as booleans and `~`/`null` as None, and `1e-3`
  without a decimal point is a *string*, not a float. Every driver validates its config
  rather than trusting it; see `load_config` in `experiments/phase0/run.py`.

---

## Tests

- `tests/` runs on CPU, no GPU, no network. **Two tiers:**

  | command | what | budget |
  |---|---|---|
  | `pytest --fast` | structural only: protocols, invariants, shapes, chunk equality, dispatch, sharding, config validation | **~2 min** |
  | `pytest` | everything except `gpu`, including the statistical and behavioural tiers | **~6 min** |

  Measured after the sharded core landed: **122 s fast** (403 tests), **310 s full** (604
  tests), CPU, 8 simulated devices, machine otherwise idle. Both are contention-sensitive: a
  concurrent job pushed the fast tier to 159 s once, which is not a regression. Re-measure
  before concluding anything from a timing.

  **The full-tier figure is stale and the budget line above is provisional (2026-08-01).** At
  645 tests it read 437 s once and 594 s once, and *both* runs shared the machine with another
  job, so the 36% spread is the contention and not the suite. The rule that applies here is
  the one directly above: neither number licenses a conclusion, so nothing was changed on the
  strength of them. Re-measure on an idle box, then update the table. Check with `uptime` and
  `ps` first, and note that the suite alone drives load past 10 on 16 cores, so a load average
  taken during a run is not evidence of an idle machine.

  **The budgets are the human constraint, not a number.** Fast has to stay usable as an inner
  loop while editing; full has to stay runnable before a commit without the urge to skip it.
  Two and six minutes are what those mean. The history is worth knowing: 90 s when the
  registry held 6 strategies, 120 s after `coupling.py`, and now this, because Phase 1 added a
  sharded core where every (strategy x device count x contraction strategy) combination
  compiles a *separate* XLA program.

  **What drives the cost is compilation, not arithmetic**, so the lever that works is running
  fewer distinct program shapes, not smaller ones. Two that paid: tiering device counts so the
  fast tier sees only 1 and 8 (the ends that matter — no collective, and full fan-out), and
  caching the D=1 reference that every invariance comparison shares. Two that did not and
  should not be retried: shrinking `n`, and dropping strategies from a parametrization.

  **Mutation testing is how a test earns its place.** `experiments/mutation.py` breaks the
  library on purpose and checks something notices. A **survivor** names a real gap; run it
  after adding tests for a new invariant, not on a schedule.

  Read "caught" as carefully as "survived". Two mutations reported caught by a `TypeError`
  and a `NameError` — both were *malformed*, leaving inconsistent shapes or naming an
  unimported symbol. A mutation that crashes proves the harness ran, not that the suite
  defends anything, and one of the two was hiding a genuine gap underneath.

  The gap it found is the one worth generalising: a test that **reimplements the logic it is
  testing** compares its own copy against itself and cannot fail.
  `test_sobol_streams_get_different_direction_numbers` restated the Sobol block arithmetic
  inline, so a mutation making every stream draw block 0 — the exact defect B1 had just fixed
  — survived. The fix was to make the code testable (`ScrambledSobol.directions` is public)
  rather than to keep the duplicate in sync.

  Record equivalent mutants in the harness rather than deleting them, so nobody retries one.

  **Every gap it has found so far was a duplicated fact**, and that is more useful than any
  individual fix. A test that restated the Sobol block arithmetic instead of calling it; the
  `1/(n·sigma)` factor living in both `estimator.estimate` and `core.tell`, with only the
  first defended. When one fact has two homes, expect the test to be guarding whichever copy
  it is nearer to. Look for the second copy.

  Cut tests when they are redundant, not when the clock is inconvenient. Move a test to the
  slow tier when it is *behavioural* rather than structural — `test_tell_descends_on_the
  _objective` runs five generations to check a sign convention, and that is exactly the shape
  of thing the full tier is for.

  The default is the complete suite. Making speed the default would mean the statistical
  tests only run when someone remembers a flag, which is the failure the old rule guarded
  against, pointed the other way.

  Mark a test `slow` when it is a **statistical measurement rather than a unit test** —
  Monte Carlo unbiasedness, bias estimates, anything whose cost is proportional to the
  confidence it buys. Do not mark something slow because it happens to be slow; fix that
  instead.

  The original rule said two minutes total. That number was written before any code
  existed and it started driving design: the alternative to this split was cutting `R` in
  the unbiasedness tests from 10 000 to 5 000, which trades a 4× margin on the 2% gate for
  2.8× to satisfy a figure nobody had measured.

  Worth recording, because it is the lesson rather than the rule: when the suite was
  first profiled the estimator file cost 42 s and that was assumed to be its Monte Carlo.
  It was not. It was the *structural* Strategy-A-vs-B chunking tests, whose cost came
  from `estimate` unrolling one traced sample/contract per chunk. Converting that to a
  `lax.scan` took the file to 29 s and fixed a real limitation for large sweeps. The tier
  split, by itself, bought 8 s. **Profile before deciding what to cut**; the expensive
  tests and the statistically expensive tests were not the same set.

  The same lesson again, larger, when coupling landed: `tests/test_coupling.py` cost 38 s,
  and `test_hd_block_is_exactly_orthogonal[8]` alone cost 1.2 s on 8×8 arrays. None of it
  was arithmetic. **A test helper that calls a strategy or a coupling outside `jax.jit`
  dispatches every primitive separately, and each dispatch compiles its own tiny HLO
  module.** A coupling is a few hundred primitives (an FWHT chain, a 30-step XOR loop, a
  scan), so per-primitive compilation dominates completely. Wrapping the helpers in `jit`
  took that file to 8 s and the fast tier from 98 s to 86 s, with no test weakened. Do it in
  new test helpers by default: `jit` is not part of any contract under test, so it costs
  nothing to add.

  `coupling._direction_numbers` is the case to remember for library code rather than tests:
  it was memoized and returned a `jnp` array, so the first call cached a value built inside
  whatever trace reached it first, and every later trace got a leaked tracer. **Anything
  memoized inside `src/` has to be host data.** No eager test can see this, because there is
  never a second trace; it showed up the moment a test helper was wrapped in `jit`.
- Multi-device tests use `XLA_FLAGS=--xla_force_host_platform_device_count=8`. Set it in
  `conftest.py` so it can't be forgotten.
- Anything needing a real GPU lives in `tests/gpu/`, is marked `@pytest.mark.gpu`, is
  deselected by default, and is run manually before each gate.
- Property-based tests where the property is clear (unbiasedness, invariance,
  structure-preservation). Golden-value tests where it isn't.
- A test that has never failed is not obviously a test. When adding one, break the code
  deliberately once and confirm it catches it.

---

## Experiments vs. tests

Different things, different rules.

**`tests/`** — fast, deterministic, CPU, assert correctness, run constantly.

**`experiments/`** — slow, produce figures and tables, GPU allowed, run rarely. Each
experiment directory contains:

```
experiments/phaseN-name/
├── config.yaml        committed BEFORE the run
├── run.py             resumable; writes one file per config as it completes
├── plot.py            regenerates every figure from the results files
├── results/           raw outputs
├── figures/
└── README.md          hardware, driver, CUDA, JAX version, commit SHA, wall-clock, spend
```

The rule from `CLAUDE.md`: **no number in any markdown file without a committed script that
regenerates it.** If a number can't be reproduced from a clean checkout, delete the number.

---

## Benchmarking

Non-negotiables, because benchmarks lie more easily than tests:

- Discard ≥3 warm-up iterations (JIT compilation).
- `block_until_ready()` on every timed result (JAX dispatch is async).
- ≥5 timed repeats; report median and IQR, never a single number.
- Assert the optimizer trajectory is identical across device counts before comparing
  timings — otherwise you're benchmarking two different computations.
- Same shapes for every method being compared.
- One roofline sanity check per headline claim. If a measured throughput exceeds what the
  hardware can do, the measurement is wrong, not the hardware.

---

## Writing

- Report negative results with the same prominence as positive ones. Half the value of
  Phase 0 is in it possibly saying no.
- Distinguish measured / derived / cited / assumed. If a number came from a paper rather
  than from a run on this hardware, say which paper.
- No "up to X×". State the configuration and the median.
- Limitations sections are written by the person who knows where the bodies are, which is
  you. A limitations section a skeptic would accept is worth more than a better number.

---

## Upstream interaction

Repeating from `CLAUDE.md` because it's the rule most easily eroded: **Claude Code does not
draft GitHub issues, PR descriptions, or review comments.** Research, explanation, code
review, and finding the relevant source lines are all in scope. The words that appear under
Andres's name on someone else's tracker are written by Andres.

JAX's `docs/contributing.md` is explicit that AI agents may not autonomously submit PRs and
that contributors should not use AI to speak for them. This repo is standalone so the
letter doesn't bind it — but the evosax modernization PRs mentioned in `docs/00-context.md`
are upstream contributions, and the same norm applies there and to any JAX-adjacent work
that follows.
