# `seed_regenerated` still does not distribute its evaluation

Found 2026-08-10, in the post-fix sweep. This is the same defect as
`diagnosis-replicated-evaluation.md`, surviving in the one strategy that fix could not
reach.

> **Fixed 2026-08-11.** `ShardedES.apply` reshapes the member axis to `(D, n/D)` and vmaps
> over it, so the thing GSPMD is asked to partition is a batch axis rather than a loop.
> `docs/proposal-scan-strategies-distribute.md` has the four options that were weighed and
> why `shard_map`, which also works, was not taken. Measured on 8 simulated CPU devices at
> `n=64`: total work across devices goes 39.6 to 50.5 ms from `D=1` to `D=8` before the fix,
> and 31.2 to 8.2 ms after. The sweep numbers below and the `seed_regenerated` rows in
> `docs/03` predate it.
>
> Everything from here down is the diagnosis as written, because the reasoning is what makes
> the fix defensible and one part of it turned out to be wrong. See "the FLOP evidence was
> not evidence" at the end.

## Symptom

Wall clock per generation does not fall with the device count. From `results-postfix/`, on
8x A100-SXM4-80GB:

```
                                  D=1       D=2       D=4    eff2    eff4
strong d=512 N=256  seed_regen/A  108.56    109.04    109.61  0.498   0.248
strong d=512 N=1024 seed_regen/A  431.58    434.79    435.17  0.496   0.248
```

Parallel efficiency is `1/D` to three digits. Every other strategy in the same sweep, on the
same node, in the same process, reaches 0.65 to 0.96:

```
strong d=512 N=1024 lowrank_r1/B   39.96     20.77       -    0.962     -
strong d=512 N=1024 iid_gaussian/B 80.54     44.24     22.97  0.910   0.877
```

`1/D` is not a slow program. It is the signature of a program that does the whole job on
every device, and it is exactly what `docs/diagnosis-replicated-evaluation.md` recorded
before `ShardedES.apply` began constraining its output.

## The measurement that settles it

Timings can be argued with. The compiled module's own cost model cannot:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  python experiments/phase2/profile.py --static --config experiments/phase2/sweep-postfix.yaml \
  --d-model 512 --population 256 --strategies seed_regenerated,lowrank_r1
```

```
  config                                        evalFLOP
  d=512 N=256 lowrank_r1/A D1->D8                  0.128
  d=512 N=256 seed_regenerated/A D1->D8            1.000
```

Under SPMD the compiled module is the per-device program, so a computation that distributes
has its FLOPs fall as `1/D`. `lowrank_r1` falls to 0.128 against an ideal 0.125.
`seed_regenerated` does not fall at all. Every device evaluates the whole population, so the
wall clock cannot fall, and `1/D` follows.

This runs on simulated CPU devices and costs nothing. Only the ratio is portable; the
absolute FLOP counts are backend dependent.

## Mechanism

`ShardedES.apply` constrains the evaluation's *output* to the member axis and lets GSPMD
propagate that backwards into the producer. That works when the producer is a `vmap`:
partitioning a batch axis is what GSPMD is for.

`SeedRegenerated.apply` does not produce its output with a `vmap`:

```python
_, out = jax.lax.scan(step, None, pert.member_ids)
```

A scan's iteration space is a sequential loop, not a batch axis. Under `AxisType.Auto` XLA
cannot partition it, so it satisfies the output constraint the only way left: gather
`member_ids`, run all `n` iterations on every device, and slice the result. The constraint is
honoured and nothing is distributed.

The correspondence across the strategies is exact, which is what makes this a mechanism
rather than a story:

| strategy | evaluation | distributes |
|---|---|---|
| `iid_gaussian` | `jax.vmap(one)(pert.eps)` | yes |
| `lowrank_r1` | `jax.vmap(one)(pert.factors)` | yes |
| `seed_regenerated` | `jax.lax.scan(step, None, pert.member_ids)` | no |

`Mirrored` wraps an inner strategy rather than replacing its `apply`, so
`Mirrored(SeedRegenerated())`, the Qiu et al. configuration named in `core.py`'s own
docstring, inherits this. Not measured yet; the sweep has no such configuration.

## Why the obvious fix is wrong

Replacing the scan with a `vmap` would distribute the work and destroy the strategy. From
`seed_regenerated.py`:

> **The scan is the point.** vmapping the regeneration would materialize all n perturbations
> again and buy nothing. `lax.scan` accumulates into one params-shaped buffer, so peak memory
> is O(|params|) whatever n is. If a profile shows an (n, ...) array here, the implementation
> has quietly become IIDGaussian with extra steps.

Storage dropping from `n * |params|` to `|params|` is the entire bet Qiu et al. make, and it
is why `sweep.yaml` can run `seed_regenerated` at `d=2048` in 0.15 GB where `iid_gaussian`
needs 99. A fix that trades that away has not fixed anything.

## The half of this that already works

`SeedRegenerated.contract` scans over exactly the same `member_ids` and *does* distribute,
because `contraction.contract_sharded` runs it under `shard_map`. Manual partitioning gives
each device its own shard, so the scan runs `n/D` iterations locally and the accumulator stays
one params-shaped buffer. The accumulator comment in that file is explicit that the scan body
and `shard_map` are two halves of one design and neither works alone.

`apply` has no equivalent wrapper. That asymmetry looks like the whole defect: the contraction
was taught to shard a scan and the evaluation was not.

## Open, and left to Andres

- Whether the evaluation should get the same `shard_map` treatment, and where the wrapper
  belongs: inside `SeedRegenerated.apply`, which knows about the scan but is deliberately
  sharding-agnostic (`sharding.AXIS_TYPE_NOTE`), or in `ShardedES.apply`, which owns
  placement but would then need to know that this strategy is scan-shaped.
- Whether a `shard_map` around the evaluation conflicts with the `x` replication
  `ShardedES.apply` now performs, since `x` must stay common across members.
- Whether `Mirrored(SeedRegenerated())` behaves the same. Expected yes, unmeasured.

## What this does not change

The sharding fix in `ShardedES.apply` is not wrong or incomplete for the strategies it
covers: three of four distribute, at 0.65 to 0.96 parallel efficiency where they previously
sat at `1/D`. This is a second, narrower instance of the same defect class, in the one path
whose producer a sharding constraint cannot reach.

## The FLOP evidence was not evidence

Added 2026-08-11, while fixing this. The section above cites
`profile.py --static` reporting per-device eval FLOPs unchanged from `D=1` to `D=8` as proof
that the evaluation is replicated. **That reasoning does not hold for a scan strategy, and it
happened to reach the right conclusion.**

XLA lowers `lax.scan` to a `while` loop, and `cost_analysis` counts the loop body once
without multiplying by the trip count. So a device scanning 8 members and a device scanning
64 report the same FLOPs. Measured at `d_model=128, n=64`: 10,308,543 against 10,308,533, a
ratio of 1.000. That figure is 1.000 whether the scan is partitioned or not, in either
direction, which makes it useless as a discriminator rather than merely imprecise.

What actually carried the diagnosis was the wall clock: flat at 431.95, 435.29, 436.43,
440.35 ms across `D=1,2,4,8` while every other strategy fell. A program whose per-device work
is unchanged cannot get faster, and it did not.

This matters beyond the bookkeeping, because the same blindness is why the defect survived
review. `tests/test_sharding.py::test_the_evaluation_distributes_across_devices` was written
after the *first* replicated-evaluation incident, specifically to prevent a second one, and
it asserts a FLOP ratio. It cannot see a scan. It also hardcoded `IIDGaussian`, so it never
looked at the one strategy where the metric fails. The test generalised along the axis the
previous bug moved along, which is the natural thing to do and was not enough.

`test_every_strategy_evaluates_only_its_own_shard` is the replacement: it asserts that each
device is handed `n/D` member ids, over every strategy, which is what "the population is
divided" means and is true or false independently of what the compiler's cost model can see.
