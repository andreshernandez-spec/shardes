# Phase 0 / E1 — estimator quality

Produces figure F5: log-log, x-axis `N/d_eff`, y-axis `1 - cos(ĝ, ∇f)`, two panels
(full-rank and rank-1), four curves each, IQR bands, vertical line at `N/d_eff = 1`.

Answers Gate G0. Spec: [`docs/01-phase0-estimator-harness.md`](../../docs/01-phase0-estimator-harness.md).

**No measurement has been run.** The harness works end to end on synthetic numbers; the
estimator it calls does not exist yet. No number goes into any markdown file in this repo
without a committed script that regenerates it.

## Running it

```bash
python run.py --list          # the grid, and what is already done
python run.py --dry-run       # synthetic numbers, exercises the whole pipeline
python run.py                 # the real sweep, once shardes.estimator exists
python run.py --limit 5       # first 5 outstanding configs, for a rehearsal
python plot.py                # figures/f5-estimator-quality.png from results/
```

`--dry-run` is the dress rehearsal doc 06 asks for: it produces publication-shaped figures
from fake numbers, so a pipeline bug surfaces before the real sweep rather than during it.
Synthetic results carry `"SYNTHETIC": true` and the figure is watermarked, because a fake
figure that looks real is worse than no figure.

## Contents

| File | |
|---|---|
| `config.toml` | the non-strategy axes. **Committed before the run**, cited by SHA in the results |
| `run.py` | resumable and idempotent; one file per config, written atomically as it completes |
| `plot.py` | regenerates F5 from `results/`, no manual steps |
| `results/` | raw outputs, one JSON per config |
| `figures/` | F5 |
| `env.json` | written by the driver, never by hand: platform, device, JAX version, commit SHA, dirty-worktree flag, wall-clock, failures |

The strategy axis is **not** in `config.toml`. It comes from
`src/shardes/strategies/registry.py`, because the rank × scheme grid is non-rectangular and
lives in one place ([docs/01 C0.5](../../docs/01-phase0-estimator-harness.md)). The commit
SHA pins it just as tightly.

## What is missing

`shardes.estimator.estimate(config, key) -> (g_hat, grad)`. The driver owns config
expansion, resume, timing, environment capture, results IO and aggregation over
replicates. The gradient-estimator math is not the driver's (CLAUDE.md ground rules), so
`run.py` imports it lazily and `--dry-run` works without it.

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
