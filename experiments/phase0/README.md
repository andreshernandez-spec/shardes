# Phase 0 / E1 — estimator quality

Produces figure F5: log-log, x-axis `N/d_eff`, y-axis `1 - cos(ĝ, ∇f)`, two panels
(full-rank and rank-1), four curves each, IQR bands, vertical line at `N/d_eff = 1`.

Answers Gate G0. Spec: [`docs/01-phase0-estimator-harness.md`](../../docs/01-phase0-estimator-harness.md).

**Nothing has been run.** This directory is a placeholder. No number goes into any markdown
file in this repo without a committed script that regenerates it.

## Contents, once it exists

| File | |
|---|---|
| `config.yaml` | the sweep grid. **Committed before the run**, cited by SHA in the results |
| `run.py` | resumable and idempotent; writes one file per config as it completes |
| `plot.py` | regenerates every figure from `results/`, no manual steps |
| `results/` | raw outputs, one file per config |
| `figures/` | F5 and the supporting shaping table |
| `env.json` | written by the driver, never by hand: platform, GPU model, driver, CUDA, JAX version, commit SHA, wall-clock, spend |

## Sweep

`N` × `rank ∈ {full, 4, 1}` × `scheme` × `shaping ∈ {none, centered_ranks}` ×
`σ ∈ {3 values}`, `R ≥ 30` replicates per configuration.

The grid is **not rectangular**. `mirrored+sobol` runs on the low-rank rows only, because
full-rank sampling is in `ℝ^{mn}` and direction-number tables stop around 20k dimensions.
`mirrored+orthogonal_hd` runs on every row and is the curve that carries the G0 comparison.

| rank | schemes |
|---|---|
| full | `iid`, `mirrored`, `mirrored+orthogonal_hd` |
| 4, 1 | `iid`, `mirrored`, `mirrored+orthogonal_hd`, `mirrored+sobol` |

Metrics: cosine similarity (headline), relative MSE, bias check, wall-clock for context.

## Where it runs

Tier T2, the local RTX 3080 (16 GB, Ampere), free, about 20 hours. The sweep is
embarrassingly serial and needs one device.

No timing claim comes out of E1, which is what makes a thermally throttling laptop GPU an
acceptable host: wall-clock per estimate is recorded for context, not as a result.
Checkpoint per configuration regardless. A 20-hour sweep should never be one run.

## Gate G0

> Do rank-1 estimator-quality curves separate across sampling schemes at `N/d_eff ≳ 1`,
> when full-rank curves at `N/d_eff ≪ 1` do not?

Yes, no, and ambiguous are all acceptable results and all three get written up. A clean
negative is a measurement nobody has published and it saves a month.
