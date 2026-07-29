# shardes

Sharded evolution strategies for JAX.

> **Status: pre-alpha, planning stage.** Nothing is implemented yet. This repository
> currently contains a plan. See `PLAN.md`.
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

## Where to start

| File | What's in it |
|---|---|
| **`PLAN.md`** | Phases, decision gates, timeline, risk register. **Read this first.** |
| `CLAUDE.md` | Instructions for Claude Code sessions; ground rules and invariants |
| `docs/00-context.md` | The two papers, the ecosystem gap, prior art on coupling/QMC for ES |
| `docs/01-phase0-estimator-harness.md` | Phase 0 — measure estimator quality against an exact oracle (1 GPU) |
| `docs/02-phase1-sharded-core.md` | Phase 1 — the library (mostly CPU-simulated devices) |
| `docs/03-phase2-benchmarks.md` | Phase 2 — scaling benchmarks (8 GPUs, 4–6 h) |
| `docs/04-phase3-coupling.md` | Phase 3 — coupled sampling, conditional on Phase 0 |
| `docs/05-paper.md` | The paper: claims, experiment matrix, figures, venue |
| `docs/06-benchmark-runbook.md` | Running the benchmarks on Kaggle, TRC, and GCP |
| `docs/compute.md` | Development-time compute; superseded for benchmarking by `06` |
| `docs/conventions.md` | Code, test, numerics, and benchmarking conventions |

---

## Quick start for development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U "jax[cuda12]" pytest      # drop [cuda12] for CPU-only

# Multi-device logic is developed on CPU with simulated devices.
XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/
```

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
