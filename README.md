# shardes

Sharded evolution strategies for JAX.

> **Status: pre-alpha. Phase 0 complete, Phase 1 next.** The estimator harness, the
> perturbation strategies, the fitness shaping and the metrics are implemented and tested
> (497 tests, CPU, ~4 min). The sharded core — the actual point of the library — is not
> written yet. See `PLAN.md`.
>
> Rename freely — check PyPI availability before committing to `shardes`.

---

## The idea

Two 2025 papers made evolution strategies work at LLM scale using **incompatible
perturbation schemes**: [Qiu et al.](https://arxiv.org/abs/2509.24372) (full-rank
perturbations regenerated from seeds, population 30) and
[Sarkar et al. / EGGROLL](https://arxiv.org/abs/2511.16652) (rank-`r` factored
perturbations that are never materialized, population up to 262,144).

There is no library that can express both. The incumbent, `evosax`, flattens every solution
to one dense vector via `ravel_pytree`, which forecloses per-matrix structured
perturbation, parameter sharding, and pytree-native ES simultaneously — and it contains no
sharding code at all.

`shardes` is an `ask`/`eval`/`tell` core where the population, the rollouts, **and the
distribution state** are sharded, solutions are never globally flattened, and the
perturbation scheme is a pluggable strategy — so both published algorithms are a
two-line diff apart.

---

## What Phase 0 found

The first decision gate asked whether **coupled sampling** (orthogonalized or
low-discrepancy perturbations, instead of i.i.d. Gaussian) improves ES gradient estimates
once low-rank perturbation drops the sampling dimension far enough that `N > d_eff`.

**It does not, and the answer is a clean negative.** Across 456 configurations at `R = 30`
replicates each, `orthogonal_hd` coupling matched the uncoupled baseline to within 1% at
every population, sigma and rank, out to `N/d_eff = 42.7`. The measurement is not a failed
treatment: a 512-member coupled block is an *exactly* orthonormal basis of `ℝ⁵¹²` where the
i.i.d. block has off-diagonal Gram entries up to 0.215. The designs are maximally different
and the estimator cannot tell them apart.

The reason is visible in the figure: estimator cosine tracks `√(N/d_ambient)` — every curve
is slope ½ on log–log, and at full rank with `N = 16384` the prediction is 0.1021 against a
measured 0.1013. If quality is fixed by how many members you draw and how large the model
is, there is nothing left for *how you choose them* to influence.

**This cost ~13 GPU-hours and saved a planned month.** Phase 3 (coupling at scale) is
dropped. The strategy abstraction stays, justified by there being two real algorithms rather
than by sample design, and stays thin — coupling is a constructor argument the sharded core
never sees.

What this does **not** show: that coupling cannot help an *optimizer*. Estimator MSE is not
a proxy for task performance — parameter-space noise is doing optimization work as Gaussian
smoothing, so a better-conditioned estimate can be a worse smoother, and the classical
QMC-for-ES results are strongest on multimodal objectives that a single transformer block is
not. That question is open (`docs/BACKLOG.md` B3).

Full answer, evidence and the incidental findings (mirroring is not a free win; its sign
flips with sigma) are in `docs/01-phase0-estimator-harness.md`. Reproduce with
`cd experiments/phase0 && python run.py && python plot.py && python gate.py`.

---

## Where to start

| File | What's in it |
|---|---|
| **`PLAN.md`** | Phases, decision gates, timeline, risk register. **Read this first.** |
| `CLAUDE.md` | Instructions for Claude Code sessions; ground rules and invariants |
| `docs/00-context.md` | The two papers, the ecosystem gap, prior art on coupling/QMC for ES |
| `docs/01-phase0-estimator-harness.md` | Phase 0 — measure estimator quality against an exact oracle (1 GPU) |
| `docs/02-phase1-sharded-core.md` | Phase 1 — the library (mostly CPU-simulated devices) |
| `docs/03-phase2-benchmarks.md` | Phase 2 — scaling benchmarks (8 GPUs, 4–6 h) |
| `docs/04-phase3-coupling.md` | Phase 3 — coupled sampling. **Dropped: G0 came back no.** Kept as the record of what was predicted |
| `docs/BACKLOG.md` | Deferred questions with what would settle each, including why Sobol degraded |
| `docs/05-paper.md` | The paper: claims, experiment matrix, figures, venue |
| `docs/06-benchmark-runbook.md` | Running the benchmarks on Kaggle, TRC, and GCP |
| `docs/compute.md` | Development-time compute; superseded for benchmarking by `06` |
| `docs/conventions.md` | Code, test, numerics, and benchmarking conventions |

---

## Quick start for development

```bash
pip install -U "jax[cuda12]"             # drop [cuda12] for CPU-only
pip install -e ".[dev,experiments]"      # the suite covers the experiment drivers too

# conftest pins JAX_PLATFORMS=cpu and 8 simulated devices, so this is just:
pytest              # everything, ~2 min: includes the statistical tier
pytest --fast       # inner loop, ~1 min: structural checks only
```

The header line reports the device count. If it says anything other than
`8 device(s), platform cpu`, stop: with a CUDA jaxlib installed jax defaults to the GPU and
reports one device, and every sharding test then passes without testing sharding.

Roughly 90% of the work needs no GPU. See `docs/compute.md` before renting anything.

---

## Non-goals

- Multi-node. Single-node multi-GPU covers every gate in the plan.
- Sharded *parameters*. ES's advantage is that every device holds the model and runs
  inference independently; sharding parameters reintroduces the communication ES avoids.
- A general-purpose replacement for evosax. This targets the sharded, large-population,
  structured-perturbation regime specifically.
- Reimplementing either paper's full experimental setup. The papers stand; this is
  infrastructure.
