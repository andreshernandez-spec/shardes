# Conventions

Short, enforceable rules. If something here turns out to be wrong, change it here rather
than working around it in code.

---

## Numerics

**Test against an exact oracle wherever one exists.** The list for this project:

| Thing | Oracle |
|---|---|
| FWHT | `jax.scipy.linalg.hadamard(n) @ x` — dense `O(n²)`, useless as an implementation, perfect as a test |
| ES estimator unbiasedness | Analytic `∇f` of a quadratic `½θᵀHθ` |
| Estimator quality generally | Backprop through a differentiable model |
| `LowRank.contract` | Explicit Python loop `Σ wₙ aₙ bₙᵀ` at small `N` |
| Low-rank → full-rank | `IIDGaussian` estimator as `r` grows |
| Scrambled Sobol | First two moments after `Φ⁻¹`; equidistribution of 2-D projections |

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

- `tests/` runs on CPU, no GPU, no network, under **two minutes** total. Enforce it; a
  slow suite stops being run.
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
