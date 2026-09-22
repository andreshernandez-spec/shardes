# shardes

Sharded evolution strategies for JAX.

Evolution strategies train a model without gradients: perturb the weights, score each
perturbed copy, move toward the copies that scored well. `shardes` is an `ask` / `apply` /
`tell` core that splits that population across devices, with the perturbation scheme as a
pluggable strategy. The two 2025 results that made ES work at language-model scale,
full-rank noise regenerated from seeds ([Qiu et al.](https://arxiv.org/abs/2509.24372))
and rank-`r` factored noise ([EGGROLL](https://arxiv.org/abs/2511.16652)), are one
constructor argument apart here, on the same mesh, shaping and update.

> **Status: 0.1, pre-alpha.** The API can still change. What is here is tested, on CPU with
> eight simulated devices, and device-count invariance has been checked on real hardware
> too: the GPU suite on two T4s, and one device against eight A100s on a 0.5B-parameter
> model.

## Install

```sh
pip install shardes                                                          # the release
pip install "shardes @ git+https://github.com/andreshernandez-spec/shardes"  # main
```

Python 3.12 or newer, JAX 0.11 or newer. On an accelerator, install JAX for it first
(`pip install -U "jax[cuda12]"` or `"jax[tpu]"`). The core needs only jax, numpy and scipy.
Extras: `shardes[models]` for loading Qwen2.5 checkpoints, `shardes[tasks]` for the MuJoCo
Playground adapter.

## Quickstart

This is `examples/quickstart.py`, and the test suite runs it. It needs no accelerator:
eight simulated CPU devices stand in, and the sharding is the same program either way.

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"   # 8 devices on a CPU

import jax
import shardes
from shardes.problems import transformer_block       # a small model that uses the seams

key = jax.random.key(0)
params = transformer_block.init(key, d_model=64)
batch = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=64, batch=8, seq=32)

mesh = shardes.make_mesh()                           # every visible device, one "pop" axis
es = shardes.ShardedES(shardes.Mirrored(shardes.LowRank(r=1)),
                       n=256, sigma=1e-2, lr=1e-2, mesh=mesh)
state = es.init(key, params)

@jax.jit
def generation(state):
    pert, state = es.ask(state)                      # a Perturbation, never a batch of params
    fitness = es.apply(transformer_block.loss, state, pert)(batch)   # one loss per member
    return es.tell(state, pert, fitness), fitness

for step in range(30):
    state, fitness = generation(state)
    if step % 10 == 0 or step == 29:
        print(f"step {step:2d}  mean loss {float(fitness.mean()):.4f}")
```

Jit the whole generation rather than stepping it eagerly; that is what lets JAX settle
device placement at trace time. `tell` descends, so hand it a loss, or the negative of a
reward.

## Both published algorithms, one argument apart

```python
# Qiu et al. 2025: full-rank noise, regenerated from seeds, small population
shardes.ShardedES(shardes.Mirrored(shardes.SeedRegenerated()), n=30, ...)

# EGGROLL (Sarkar et al. 2025): rank-1 factors, never materialized, huge population
shardes.ShardedES(shardes.Mirrored(shardes.LowRank(r=1)), n=262_144, ...)
```

| strategy | the perturbation | what it costs |
|---|---|---|
| `IIDGaussian()` | full rank, materialized | memory: `n` copies of the noise |
| `SeedRegenerated(chunk=1)` | full rank, regenerated from each member's seed | compute: every perturbation is drawn twice. `chunk` trades memory for speed |
| `LowRank(r)` | rank `r`, two thin factors per matrix, product never formed | the noise is not full rank |
| `Mirrored(inner)` | antithetic pairs around any of the above | halves the distinct directions |

`ask` returns a `Perturbation`, not a batch of parameter trees, and the rest follows from
that: under `LowRank` it is a pair of factors, under `SeedRegenerated` a key and member
ids. Shaping is `centered_ranks` by default (`shardes.shaping` has `centered`,
`group_relative` and `none`).

## Where the update is assembled: `how="A"` or `how="B"`

After evaluation each device holds the fitnesses of its own members, and the update
`sum_i w_i eps_i` has to be put together. There are two placements, and which is faster
depends on the perturbation and on the interconnect.

- **`how="B"`, the default.** Each device contracts its own members, then the partial
  updates are all-reduced. That moves a buffer the size of the model.
- **`how="A"`.** Every device gathers the scalar fitnesses and contracts the whole
  population itself. Only scalars cross the wire, and the contraction is repeated on
  every device.

Measured on an 8x A100 node, a TPU v5e-8 and two A100 nodes over sockets: on one host
with a fast interconnect, **B wins for full-rank and seed-regenerated perturbations, and A
wins for low-rank perturbations on large models**, by more on the TPU. Across a slow host
boundary A wins more widely. The measurements, and the paper they belong to, are in
[shardes-paper](https://github.com/andreshernandez-spec/shardes-paper).

## Your own model

A low-rank perturbation is never materialized, so a perturbed weight is not an array: it
is a base matrix plus two factors. A model therefore routes its parameterized matrix
multiplies and embedding lookups through two seams, `shardes.nn.dense(x, w)` and
`shardes.nn.embed(table, ids)`, which do the right thing for a plain array and for a
structured weight alike. Direct arithmetic on a structured weight raises rather than
silently densifying. `shardes.problems.transformer_block` is the small worked example and
`shardes.problems.qwen2` is Qwen2.5 ported this way.
`shardes.check.check_model(model, params, batch)` finds, in a second on CPU, every weight a
model reaches without a seam, instead of the strategy raising minutes into a run.

## What it guarantees

- **Device count cannot change the result.** Member `i`'s noise derives from `i` alone,
  so the update contracted on one device and on eight agrees to floating-point
  tolerance. This is tested on simulated devices and checked on real ones: `tests/gpu`
  on two T4s (`validation/`), and on A100s the update for Qwen2.5-0.5B computed on one
  device and on eight agrees to 6.3e-6 relative error
  (`shardes-paper:experiments/countdown/results/c6d-a100x8-2026-08-18`).
- **The low-rank path never forms an `(n, m, n)` array.** A test inspects the traced
  program to make sure.
- **Fitness is float32 or wider.** In bfloat16 a population's losses collapse to a
  handful of ties and rank shaping of ties is noise, so `tell` refuses rather than casts.
- **No global flattening.** Parameters keep their shapes from sampling to update, which
  is what makes per-matrix structure expressible at all.

## Development

```sh
pip install -e ".[dev]"
pytest --fast        # the inner loop
pytest               # everything, including the statistical tier
```

The suite pins JAX to the CPU and simulates eight devices, so it needs no accelerator and
no network. `tools/` holds scripts that check the library itself (mutation testing,
compile-cost diagnostics, two probes behind design decisions), `validation/` the
real-hardware invariance check, and `docs/` the design, conventions and diagnoses.
CI runs the suite on Python 3.12 and 3.13 and installs the built wheel into a clean
environment.

## The paper and the experiments

This library is the instrument of *Update-contraction placement in sharded evolution
strategies on GPUs and TPUs*. The manuscript, every experiment and every result live in
[shardes-paper](https://github.com/andreshernandez-spec/shardes-paper), which installs
this library at a pinned commit.

Until the tag `monorepo-final` the two were one repository, and that history is kept here
on purpose: result records from that period cite commits of this repository, and
checking one out gives the driver, its config and the library together, as run.

## Non-goals

- Sharded *parameters*. Every device holds the model and evaluates independently;
  sharding parameters would bring back the communication ES avoids.
- A general replacement for evosax. This targets the sharded, large-population,
  structured-perturbation regime.
- Reimplementing either paper's full experimental setup. The papers stand; this is
  infrastructure.

## License

Apache 2.0.
