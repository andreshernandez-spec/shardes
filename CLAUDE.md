# CLAUDE.md

Instructions for Claude Code working in this repository. Keep this file short — it
loads into every session. Detail lives in `docs/`.

---

## What this project is

`shardes` is a JAX library for **sharded evolution strategies**: an `ask`/`eval`/`tell`
core that shards the population and the rollouts across devices, with the perturbation
scheme as a pluggable, shape-aware strategy rather than a hardcoded `randn(num_dims)`.

The distribution state is replicated, not sharded (decided in `docs/02` C1.4 and corrected
here on 2026-07-31). Sharding it is theatre for isotropic ES and, for the CMA family, would
cost a per-generation gather to save memory the design already spends on replicating the
model.

It exists because the two 2025 papers that made ES work at LLM scale
([Qiu et al.](https://arxiv.org/abs/2509.24372), full-rank + seed regeneration;
[Sarkar et al. / EGGROLL](https://arxiv.org/abs/2511.16652), rank-`r` factored) have no
common library, and the incumbent (`evosax`) forecloses both by flattening every solution
to one dense vector via `ravel_pytree`. Background: `docs/00-context.md`.

**The deliverable is the library.** Everything else is instrumentation for it.

---

## Ground rules

### 1. Andres writes and understands the load-bearing code

This is a hard constraint, not a preference. The project's value is what he can explain
in a technical interview, and the ecosystem this targets is explicit about it — JAX's
`docs/contributing.md` states that *"the use of an AI agent that autonomously writes code
and submits pull requests is not permitted"* and *"do not use AI to speak for you"* on
issue trackers. This repo is standalone, so the letter of that policy doesn't bind it,
but the spirit is the whole point.

Concretely:

- **Claude Code may**: scaffold, write tests, write benchmark harnesses, write plotting
  and infra code, port reference implementations for comparison, research, review, explain,
  find bugs, refactor.
- **Claude Code should hand back**: the perturbation strategies, the sharding logic, the
  gradient-estimator math, and anything that ends up in a README or a talk. Draft it,
  explain the reasoning, then let Andres write the version that ships.
- **Claude Code must never**: draft GitHub issues, PR descriptions, or review comments for
  upstream projects. Andres writes those in his own voice.

If asked to do something in the second or third bucket, say so and offer the first-bucket
version instead.

### 2. Every number in a doc has a script behind it

No claim of the form "X is Y× faster" or "error is 1e-15" enters any markdown file without
a committed, re-runnable script and a recorded environment (GPU model, JAX version, commit
SHA). Scripts that check the library are in `tools/`; the experiments and the paper are in
the [shardes-paper](https://github.com/andreshernandez-spec/shardes-paper) repository, and a
number that comes from there is cited as `shardes-paper:<path>`. If a number can't be
reproduced from a clean checkout, delete it.

### 3. Don't build what a measurement already ruled out

Phase 3, coupled sampling at scale, was conditional on a Phase 0 result and that result
came back no (`docs/04-phase3-coupling.md`, `docs/01`). Do not build it speculatively
because it seems interesting. The research program's phases and gates are in
`shardes-paper:PLAN.md`.

---

## Repository layout

```
shardes/
├── README.md              for a user: install, quickstart, guarantees
├── CLAUDE.md              this file
├── pyproject.toml         must stay pip-installable from a git SHA; shardes-paper pins one
├── src/shardes/           the library
├── tests/                 pytest; CPU-only, no network; tests/gpu needs real accelerators
├── examples/              what the README shows; the suite runs them
├── tools/                 scripts that check the library, not experiments
├── validation/            the real-hardware invariance check (gate G1)
└── docs/                  design (02), context (00), the estimator study (01), conventions,
                           diagnoses, proposals, and the record of the split (14)
```

The experiments, the results, the paper and the campaign docs are in
[shardes-paper](https://github.com/andreshernandez-spec/shardes-paper), which installs this
library at a pinned commit. Until the tag `monorepo-final` they were here, and that history
is kept on purpose: records from that period cite commits of this repository. Never rewrite
it, and never delete a `provenance/*` tag: each one keeps a cited commit reachable. It was
rewritten once, on 2026-09-22, to remove a personal account from three files; every commit
after 2026-08-01 changed hash and `docs/provenance/` maps old to new. That is the exception,
and the map is what it cost.

`tests/` runs on CPU, no GPU, no network, in **two tiers**: `pytest --fast` is the inner
loop while editing, `pytest` is everything and is the default. Budgets and the reasoning
are in `docs/conventions.md`. This line said "under two minutes" until 2026-08-01; that
number predated the code and was about to cost `R` in the unbiasedness tests.

---

## Environment

- Python ≥ 3.12 (jax 0.11 requires it; this said 3.11 until CI tried it), JAX ≥ 0.11. **Do not pin below 0.11** — `from jax import shard_map` needs
  0.8 and `AxisType` needs 0.11, and those are what the library is built on. This used to
  say "evosax is stuck at `<0.7`"; that stopped being true at evosax 0.2.0 (`jax>=0.5.0`,
  no upper bound). The floor is justified by what we use, not by what they pin.
- `from jax import shard_map`. `jax.experimental.shard_map` is deprecated (JAX 0.8.0).
- Multi-device logic is developed and tested **on CPU** with
  `XLA_FLAGS=--xla_force_host_platform_device_count=8`. See `docs/compute.md`. Do not
  reach for a GPU to debug a `PartitionSpec`.
- Dependencies stay minimal: `jax`, `numpy`, `pytest`. Adding anything else needs a
  one-line justification in the PR description. `optax`, `flax`, `chex` are acceptable if
  actually used; `evosax` is a comparison target, not a dependency.

---

## Invariants — breaking these is a bug, not a tradeoff

1. **No global flattening.** Nothing in `src/` calls `ravel_pytree` on a solution. Leaves
   keep their `(m, n)` shape all the way through sample → apply → contract. This is the
   architectural difference from evosax and the reason low-rank perturbation is expressible
   at all.
2. **Device-count invariance.** For a fixed seed, the update produced on 1 device and on
   8 devices must agree to within float tolerance (`rtol=1e-5` bf16, `1e-12` f32). Seeds
   are derived from the *member index*, never from the device index. There is a test for
   this; it is the most important test in the repo.
3. **The perturbation is never materialized by the low-rank path.** If a profile shows an
   `(n_members, m, n)` array being allocated under `LowRank`, the implementation is wrong.
4. **Communication is measured, not assumed.** Every `psum`/`all_gather` in the update path
   is accounted for in `shardes-paper:docs/03-phase2-benchmarks.md`. See the note there about the two
   contraction strategies — the "ES only all-reduces scalars" claim is true only for one
   of them.

---

## Working style

- Small commits, each with its test. Prefer a failing test first.
- When a design decision has two defensible answers, write both down in the relevant
  `docs/` file with the tradeoff, and flag it for Andres rather than picking silently.
- Numerical code: assert against an exact oracle where one exists (see
  `docs/conventions.md` for the list — e.g. the FWHT has one that ships in JAX).
- The names in `shardes.__all__` are the supported surface and `tests/test_public_api.py`
  pins them. Changing one is a version bump and a line in the release notes. Everything
  else stays importable from its module, and shardes-paper uses those deep paths, so a
  rename there breaks the paper's pin when it next moves.
