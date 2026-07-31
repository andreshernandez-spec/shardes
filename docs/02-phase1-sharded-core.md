# Phase 1 — The sharded ES core

**Compute**: CPU with 8 simulated devices for ~90% of it, plus 1–2 real GPUs to confirm.
**Duration**: 6–10 weeks. This is the project.
**Gate**: G1 — see the bottom of this file.

---

## Goal

An `ask` / `eval` / `tell` ES core where:

- the population is sharded across devices,
- rollouts are sharded across devices,
- the distribution state is *replicated*, and C1.4 explains why that is the
  right answer rather than a shortfall,
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

### C1.4 — Distribution state — **DECIDED 2026-07-31: isotropic, plus a diagonal**

For isotropic ES the state is `(μ, σ)` and sharding it is theatre — say so honestly rather
than dressing it up.

Where it matters is the CMA family, where the `d×d` covariance is replicated on every
device in every existing JAX implementation. The options were:

- **Option 1**: carry one CMA-family strategy with genuinely sharded state (VD-CMA is the
  best candidate — diagonal-plus-rank-one, so the state shards naturally and it's verified
  absent from evosax).
- **Option 2**: ship isotropic only, and state the limitation in the README.

**Decision: Option 2, widened to a per-coordinate diagonal. Option 1 is dropped.**

Three findings drove it, and the third is the one that settles it.

**1. The memory argument for Option 1 is weaker than this document assumed.** Parameters are
`O(d)` and *deliberately replicated* — see the traps below: every device holding the model is
ES's whole advantage. VD-CMA's state is diagonal-plus-rank-one, also `O(d)`. If a device can
hold the params it can hold the state, so sharding it saves a small constant multiple of
something already chosen to be replicated. The memory case is strong only for full CMA's
`O(d²)`, and full CMA is a non-starter at this scale regardless: `d = 10⁹` means `10¹⁸`
covariance entries.

**2. Sharding an `O(d)` state on a population-parallel mesh costs a gather.** The mesh has
one axis. To sample, device `k` needs the distribution parameters for every leaf it perturbs,
which is all of them. Sharding `D` and `v` across `pop` therefore forces an all-gather every
generation — precisely the `O(d)` cost Strategy B pays. It trades replicated memory for
recurring communication rather than removing a cost.

**3. Option 1 requires a protocol change, which cuts against what Gate G0 concluded.**
`sample(base_key, params, member_ids)` has nowhere to put covariance state; the only
distribution parameter the protocol carries is `sigma`, at `apply`. VD-CMA's rank-one term
couples coordinates, so it cannot be expressed as a per-leaf scale — it would need a fourth
protocol method or state threaded through `sample`. G0's finding was that the strategy
abstraction is **not** load-bearing for sample design and should stay thin; widening it for a
covariance family is the opposite move, made for a claim whose memory justification is
already the weakest of the three.

There is also a composition problem: EGGROLL's factored perturbation already determines the
sampling distribution, and so does VD-CMA. Two components owning one decision is a design
conflict, not an integration task.

**What ships instead.** `sample` produces **unit-scale** perturbations and `apply` scales
them, so widening `sigma` from a scalar to a params-shaped pytree gives per-coordinate step
sizes with **no protocol change**: it is a type widening on an argument that already exists.
The diagonal shards exactly as params do, it composes with `LowRank` because it is a scale
rather than a distribution, and it buys the thing isotropic ES is genuinely bad at, which is
ill-conditioning.

Not claimed as novel. VD-CMA's absence from evosax was verified; the diagonal families
(sep-CMA, SNES) were **not** checked, and the reason to do this is that it fits the protocol
for free, not that nobody else has it.

**The diagonal is supported, not learned, and that is deliberate.** `sigma` is whatever the
caller set at `init`, for every generation. Making it *adapt* turns out to need a second
moment of the perturbation — `Σ uᵢ(εᵢ² − 1)` for SNES, `Σ wᵢyᵢ²` for sep-CMA's rank-μ update —
and `contract` computes `Σ wᵢεᵢ` and is linear in the weights, so no choice of weights
produces it. CSA is not a way out: it adapts a *scalar* step size from the evolution path, and
there is no published rule that gets a per-coordinate sigma from the mean shift alone.

Deferred rather than built, under PLAN.md ground rule 3: no gate needs it, none of the paper's
claims rest on it, and G0's finding was that this abstraction should stay thin. The design and
the trigger are in `docs/BACKLOG.md` B6, costed at about half a day when a benchmark turns out
to be ill-conditioned enough to need it.

**The consequence for the project's headline claim.** "The distribution state is sharded" was
in `README.md`, `CLAUDE.md` and `PLAN.md`, and it is not supportable for *any* ES that keeps
parameters replicated — which this one does, on purpose. All three now say what is actually
true: the population and the rollouts sharded, the distribution state replicated because it
is the same size as the model and the model is replicated by design. Retracting a claim is
cheaper than defending one that does not hold.

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

**Installed and verified 2026-07-31.** `pip install playground` — MuJoCo Playground 0.2.0,
MuJoCo 3.11.0, MJX 3.11.0. Notes worth keeping:

- **It is free, and the licence question is stale.** MuJoCo required a paid Roboti licence
  until DeepMind acquired it in 2021; it has been Apache 2.0 since 2022. No key, no account.
- **Nothing in the chain can force a JAX downgrade**, which was the real risk and the reason
  to check rather than assume. Every constraint is a *lower* bound: `brax jax>=0.4.6`,
  `flax jax>=0.10.0`, `mujoco-mjx` unpinned. A `pip install --dry-run` confirmed the resolver
  does not touch `jax` or `jaxlib`. That is exactly the trap evosax is in with `jax<0.7`, so
  it was worth one command to be sure.
- **This paragraph's own reasoning is half wrong and the correction matters.** "Use MuJoCo
  Playground *because* brax.envs is deprecated" does not escape brax: Playground depends on
  `brax>=0.14.2`, which pulls `jaxopt` (last release April 2025, folded into optax). The
  honest version is that Playground is a *maintained wrapper around* brax rather than an
  alternative to it. Both were verified to import and run under JAX 0.11, so the risk did not
  materialise, but it is a live dependency to watch rather than one that was avoided.
  Building on `mujoco-mjx` alone would have avoided brax entirely at the cost of writing the
  environment loop; Playground was chosen for its 54 ready-made envs.
- **Verified by running, not by importing.** `CartpoleBalance` reset/step under `jit`, and an
  8-member `vmap` of a 20-step `lax.scan` rollout — which is the shape ES actually needs, and
  the only one that would have exposed a vmap or scan incompatibility. Returns peaked at zero
  action scale with symmetric falloff, which is what balancing a cartpole should look like.
- MuJoCo 3.11 runs its physics through `mujoco_warp` (NVIDIA Warp), which JIT-compiles its
  own kernels on first use. First `reset` cost ~14 s of compilation on CPU. Budget for that
  in any timing harness; it is not per-step cost.

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
