# shardes

Sharded evolution strategies for JAX.

> **Status: pre-alpha. Phase 0 complete; Phase 1 capabilities complete, Gate G1 not yet
> closed.** The estimator harness, the perturbation strategies, the couplings, the shaping,
> the sharded `ask`/`tell` core, both contraction strategies and a MuJoCo Playground task
> adapter are implemented and tested (630 tests, CPU, ~6 min).
>
> Device-count invariance — the property everything else is built to permit — passes on 1, 2,
> 4 and 8 simulated devices for every strategy and both contraction strategies, and one real
> GPU reproduces the simulated result. **The two-GPU confirmation is the one thing G1 still
> wants**, and it is deliberately not fakeable on this hardware: simulated devices share a
> memory space and never actually communicate. See `docs/06` T2′.
>
> Phase 2 (scaling benchmarks) has not started. Nothing here is a timing claim.
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

`shardes` is an `ask`/`eval`/`tell` core where the population and the rollouts are sharded,
solutions are **never globally flattened**, and the perturbation scheme is a pluggable
strategy — so both published algorithms are a two-line diff apart.

The distribution state is *replicated*, deliberately. An earlier version of this claimed it
was sharded; that is not supportable for any ES that keeps parameters replicated, which this
one does on purpose, because every device holding the model is the advantage ES has over
gradient training. `docs/02-phase1-sharded-core.md` C1.4 has the full reasoning and what
shipped instead.

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

## Both published algorithms, one argument apart

The architectural claim, made concrete. These differ in the `strategy=` line and nothing
else — same core, same shaping, same sharding, same contraction:

```python
from jax import make_mesh
from shardes import sharding
from shardes.core import ShardedES
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

mesh = sharding.make_mesh()          # every visible device, one "pop" axis

# Qiu et al. 2025 — full-rank perturbations regenerated from seeds, small population
qiu = ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, sigma=0.01, lr=0.05, mesh=mesh)

# EGGROLL (Sarkar et al. 2025) — rank-1 factored, never materialized, huge population
eggroll = ShardedES(strategy=Mirrored(LowRank(r=1)), n=262_144, sigma=0.01, lr=0.05, mesh=mesh)
```

Driving either one is the same three calls. Jit the whole generation rather than stepping
it eagerly — that is what lets JAX settle device placement at trace time:

```python
state = eggroll.init(key, params)

@jax.jit
def generation(state, batch):
    pert, state = eggroll.ask(state)          # a Perturbation, never materialized params
    fitness = eggroll.apply(model, state, pert)(batch)
    return eggroll.tell(state, pert, fitness)
```

`ask` returning a `Perturbation` rather than a batch of parameter trees is the decision the
rest follows from: under `LowRank` the thing it returns is a pair of factors whose product
is never formed, and under `SeedRegenerated` it is a key and a set of member ids and no
noise at all. A library that hands back materialized trees cannot express either.

`tell` **descends** on what it is given, so a reward gets negated first.

One constraint, and it is the first thing you will hit: **the model's matmuls go through
`shardes.nn.dense`**, not `x @ W.T`. `LowRank` perturbs by substituting a structured weight
into the params tree, and a model that does arithmetic on that weight directly raises rather
than silently computing something else. That single indirection is what makes low-rank
perturbation expressible without a jaxpr interpreter, and it is the cost of it:

```python
from shardes.nn import dense

def model(params, x):
    h = jnp.tanh(dense(x, params["w1"]) + params["b1"])
    return jnp.sum(dense(h, params["w2"]))          # not h @ params["w2"].T
```

Embeddings need the same treatment through `embed`, because an embedding is a *gather*
rather than a matmul and `dense` cannot see it:

```python
from shardes.nn import dense, embed

h = jnp.mean(embed(params["emb"], token_ids), axis=1)   # not params["emb"][token_ids]
logits = dense(h, params["out"])
```

That is what makes embedding tables expressible under low-rank perturbation at all —
`(E + sAB^T)[ids] = E[ids] + sA[ids]B^T`, so the table is never formed. EGGROLL's reference
implementation raises `NotImplementedError` on this path (`docs/BACKLOG.md` B4).

`IIDGaussian` and `SeedRegenerated` substitute ordinary arrays and take the array branch of
both seams, so this only binds if you want the low-rank path.

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
