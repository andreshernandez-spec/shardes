# Phase 1 — The sharded ES core

**Compute**: CPU with 8 simulated devices for ~90% of it, plus 1–2 real GPUs to confirm.
**Duration**: 6–10 weeks. This is the project.
**Gate**: G1 — see the bottom of this file.

---

## Goal

An `ask` / `eval` / `tell` ES core where:

- the population is sharded across devices,
- rollouts are sharded across devices,
- **the distribution state is sharded, not replicated**,
- solutions are **never globally flattened** — leaves keep their `(m, n)` shape,
- the perturbation scheme is a pluggable strategy, so both published algorithms are two
  lines of config apart.

The first four are what evosax doesn't do. The fifth is what makes the first four useful.

---

## Capabilities delivered

### C1.1 — Pytree-native ask/tell

```python
state = es.init(key, params)              # params is a pytree; no ravel_pytree anywhere
pert, state = es.ask(key, state, n=N)     # shape-aware perturbation for N members
fitness = evaluate(params, pert)          # user-supplied; shape (N,)
state = es.tell(state, pert, fitness)
```

`ask` returns a `Perturbation`, not a batch of parameter trees. This is the single most
consequential API decision: returning materialized parameter trees would make `LowRank`
inexpressible, which is exactly the trap evosax fell into.

### C1.2 — Mesh, sharding, and the seed contract

```python
from jax import shard_map                    # NOT jax.experimental.shard_map (deprecated 0.8.0)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

mesh = Mesh(mesh_utils.create_device_mesh((n_devices,)), axis_names=("pop",))
```

- Parameters: **replicated** across `pop`. Every device holds the full model. For ES this
  is correct and cheap — it's the perturbations and rollouts that need splitting.
- Perturbations: sharded on the member axis, `P("pop", None, None)`.
- Fitnesses: sharded `P("pop")`.
- Distribution state: see C1.4.

**The seed contract**, which everything else depends on:

> Member `i`'s perturbation is derived from `jax.random.fold_in(base_key, i)` where `i` is
> the **global member index**. Never the device index, never a per-device counter.

This is what makes device-count invariance possible, and it's also what makes Qiu-style
seed regeneration work at all. Enforce it in code review and in `test_device_invariance`.

### C1.3 — Two contraction strategies, both implemented and both measured

**This is the most interesting systems question in the project. Do not pick one silently.**

The claim "ES only needs to all-reduce scalar fitnesses" is repeated a lot. It is true for
strategy A and false for strategy B, and both are legitimate.

**Strategy A — scalar all-reduce, replicated regeneration.**
All-reduce the `N` fitness scalars (and, if not derivable, the `N` seeds — a few KB). Every
device then regenerates all `N` perturbations from seeds and contracts locally.
- Communication: `O(N)` scalars. Tiny.
- Compute: contraction is replicated `D` times. Wasteful, but cheap next to rollouts.
- Requires: seed-derivable perturbations (all three strategies satisfy this).

**Strategy B — model-size all-reduce of the partial update.**
Each device contracts only its local shard into a full params-shaped partial, then `psum`
over devices.
- Communication: `O(d)` — one model-sized all-reduce per generation, same as data-parallel
  SGD.
- Compute: contraction is split `D` ways.

Crossover depends on `N`, `d`, `D`, and interconnect. Both get implemented; Phase 2
measures the crossover and produces the plot. **Nothing public claims "scalar all-reduce"
until that measurement exists.**

### C1.4 — Sharded distribution state

For isotropic ES the state is `(μ, σ)` and sharding it is theatre — say so honestly rather
than dressing it up.

Where it matters is the CMA family, where the `d×d` covariance is replicated on every
device in every existing JAX implementation. Decide in this phase:

- **Option 1**: carry one CMA-family strategy with genuinely sharded state (VD-CMA is the
  best candidate — diagonal-plus-rank-one, so the state shards naturally and it's verified
  absent from evosax).
- **Option 2**: ship isotropic only, and state the limitation in the README.

Option 1 is the stronger claim and roughly two extra weeks. Option 2 is defensible. Pick
deliberately, write down why.

### C1.5 — Both algorithms as strategies

```python
# Qiu et al.
es = ShardedES(strategy=Mirrored(SeedRegenerated()), n=30, sigma=...)

# EGGROLL
es = ShardedES(strategy=Mirrored(LowRank(r=1)), n=262_144, sigma=...)
```

If switching between the two published algorithms is not close to this, the abstraction
failed and should be reworked before going further.

### C1.6 — Fitness shaping, including the group-relative variant

Centered ranks (standard), plus **group-relative / GRPO-style shaping** — the thing that
makes ES competitive with GRPO on reasoning, verified absent from evosax's shaping module.
Small surface, high leverage.

Note: centered ranks need a global sort over all `N` fitnesses, which is an `all_gather` of
`N` scalars plus a synchronization barrier. Cheap in bytes, not free in latency. Measure it
in Phase 2.

### C1.7 — A task adapter that isn't deprecated

evosax's problem adapters target `brax.envs`, deprecated at Brax v0.13.0. Use MuJoCo
Playground. One small env is enough for Phase 1; Phase 2 needs something that actually
saturates 8 GPUs.

---

## How to test it

`tests/` stays CPU-only and under two minutes. The trick that makes this possible:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/
```

Eight simulated devices on CPU. Sharding logic, `PartitionSpec` errors, `shard_map`
signatures, collective placement — all reproduce faithfully. **Do not rent a GPU to debug
a sharding annotation.**

| Test | Asserts |
|---|---|
| `test_device_invariance` ⭐ | Same seed on 1 device and on 8 simulated devices → same update. `rtol=1e-12` f32. **The most important test in the repo.** |
| `test_no_ravel_pytree` | Static check: no `ravel_pytree` import or call anywhere under `src/` |
| `test_strategy_A_equals_strategy_B` | The two contraction strategies produce the same update for the same seed |
| `test_comm_volume_A` | Instrument collectives; assert Strategy A moves `O(N)` not `O(Nd)` |
| `test_comm_volume_B` | Assert Strategy B moves exactly one params-sized `psum` per generation |
| `test_seed_by_member_index` | Member `i`'s perturbation is identical regardless of `n` or device count |
| `test_ask_tell_roundtrip` | Sphere/Rastrigin: ES actually descends, for every strategy |
| `test_state_sharding` | Distribution state has the intended `NamedSharding`, not replicated |
| `test_pytree_structure_preserved` | Update tree structure == params tree structure, leaf shapes match |
| `test_lowrank_matches_reference` | Against a naive materialize-everything implementation, small `m,n,N` |

GPU-only, run manually, not in CI: a 1-GPU and a 2-GPU run reproducing
`test_device_invariance` against the CPU-simulated result. This is the check that the
simulated-device shortcut didn't lie to you. Do it **before** Phase 2, not during.

---

## How to showcase it

Three artifacts:

1. **The two-line diff.** A README snippet showing Qiu-style and EGGROLL-style ES
   configured from the same API. This is the architectural claim, made concrete.
2. **A memory plot.** Peak memory per device vs. device count, showing state and
   perturbation storage falling as `1/D` while a replicated baseline stays flat. Runs on
   simulated devices for the accounting; confirm on real GPUs.
3. **A communication table.** Bytes moved per generation, per strategy, as a function of
   `N`, `d`, `D` — analytic prediction next to instrumented measurement. If they disagree,
   you've found a bug, which is the point.

Plus: the test suite itself is a showcase. `test_device_invariance` passing on 8 simulated
devices is a thing a reviewer can run in thirty seconds.

---

## Exit criteria — Gate G1

All of:

1. `pytest tests/` green on CPU with 8 simulated devices, under two minutes.
2. `test_device_invariance` passes on CPU-8, and is reproduced on a real 2-GPU box.
3. Both published algorithms run end-to-end on a MuJoCo Playground task from the same API.
4. Both contraction strategies implemented, and their communication volume instrumented
   and matching the analytic prediction.
5. The distribution-state decision (C1.4) is made and written down with its rationale.
6. No `ravel_pytree` under `src/`.

---

## Traps

- **`jax.experimental.shard_map` is deprecated.** Use `from jax import shard_map`. EGGROLL's
  own scripts use the deprecated path; don't copy them.
- **Fitness shaping is a barrier.** Global rank computation synchronizes every device every
  generation. Fine, but know it's there before the scaling curve surprises you.
- **Don't shard parameters.** Tempting, wrong for this workload. ES's advantage is that
  every device can hold the model and run inference independently; sharding parameters
  reintroduces exactly the communication ES avoids. If the model doesn't fit, that's a
  different project.
- **bf16 accumulation in `contract`.** Summing `N = 2¹⁸` terms in bf16 loses precision
  badly. Accumulate in f32 even when the perturbations are bf16, and test it.
- **Simulated devices don't model interconnect.** They validate correctness and byte
  counts, never latency. Every timing claim needs real hardware.
