# Why the evaluation is replicated instead of sharded

**Status: diagnosis plus the fix, drafted by Claude Code for Andres to review and rewrite.**
The constraint described under "What the fix looks like" is now in `ShardedES.apply`, and
`tests/test_sharding.py::test_the_evaluation_distributes_across_devices` holds it. This is
sharding logic, which ground rule 1 puts in Andres's hands; it was written here on his
explicit instruction and still wants reading line by line.

Everything below is measured, and every measurement runs on CPU with
`XLA_FLAGS=--xla_force_host_platform_device_count=8`, because all of it is a property of the
compiled program rather than of hardware.

---

## The symptom

`experiments/phase2/profile.py --static` reports the FLOPs of the compiled per-device
program. Under SPMD that module is what one device executes, so a computation that
distributes has them fall as `1/D`. In all 16 sweep configurations they do not move at all:

```
d=512  N=1024  iid_gaussian/A   D1->D8   evalFLOP 1.000
d=2048 N=256   seed_regenerated/B D1->D8 evalFLOP 1.000
... 16 of 16
```

The HLO says the same thing without the abstraction. At `d=512, N=1024` the three dominant
`dot` instructions have leading dimension **1024 at both `D=1` and `D=8`**. At `D=8` a
partitioned program would show 128.

This accounts for M1 exactly: if per-device work does not fall, `T_D = T_1` and parallel
efficiency is `1/D`. Measured 0.112 to 0.142 at `D=8` against `1/8 = 0.125`.

---

## The cause

The mesh is `AxisType.Auto` (`sharding.py`, and the reasoning in `AXIS_TYPE_NOTE` is sound:
Explicit would push mesh-aware annotations down into every strategy). Auto means GSPMD
*propagates* sharding outward from the places the program constrains it. Nothing is
declared; everything is inferred.

Follow what is actually constrained:

| where | what | to |
|---|---|---|
| `sharding.member_ids` | the `arange(n)` of member indices | `P("pop")`, sharded |
| `core.py` `tell` | **fitness** | `replicated` |
| `core.py` `tell` | shaped weights | `P("pop")` |
| `contraction.py` | ids and weights, strategy A | `replicated` |

The fitness is constrained exactly once, in `tell`, and it is constrained to **replicated**,
because `centered_ranks` needs a global sort. That is correct for the sort and it is the
only statement the program ever makes about how the fitness is laid out.

GSPMD propagates that constraint *backwards*. The cheapest way to have the whole fitness
vector on every device is to compute the whole fitness vector on every device, and doing so
requires no collective at all. So the compiler replicates `apply`, and behind it `sample`,
and the sharded `member_ids` gets gathered early and ignored.

**Nothing downstream ever asks for a sharded fitness, so nothing pushes back.** The sharded
`member_ids` is a producer-side hint, and a producer-side hint loses to a consumer-side
constraint.

---

## Sharding the perturbation does not fix it

The obvious suspicion is that the perturbation is not placed on the member axis. Measured,
with `shard_perturbation` applied to `pert.eps` before `apply`:

```
as written             D1= 985,741,918,208  D8= 985,741,918,208   D8/D1=1.000
+shard eps             D1= 985,741,918,208  D8= 985,741,918,208   D8/D1=1.000   <- no change
+constrain fitness     D1= 985,741,918,208  D8= 123,221,868,544   D8/D1=0.125   <- exactly 1/8
```

Constraining the input does nothing because the consumer is free to gather it and compute
everywhere, which is what it does. Constraining the *output* forces the `vmap` to be
partitioned, and that back-propagates to shard `eps` on its own.

This also explains a piece of dead code. `sharding.shard_perturbation` and
`sharding.per_member` are called **only from `tests/test_sharding.py`** and from nowhere in
`src/`. They were written for exactly this job and they constrain the wrong end, so the
suite reads as though the perturbation is sharded while the pipeline never places it.

---

## What the fix looks like

One constraint, where the fitness is produced rather than where it is consumed. Roughly:

```python
# ShardedES.apply, on the value the returned g(x) hands back
return jax.lax.with_sharding_constraint(fitness, sharding.members(self.mesh))
```

Measured on the full generation, `d=512, N=1024`:

| | `D8/D1` FLOPs |
|---|---|
| strategy A, as written | 1.000 |
| strategy A, fitness sharded at production | 0.258 |
| strategy B, as written | 0.972 |
| strategy B, fitness sharded at production | **0.125**, exactly `1/8` |

B reaches perfect strong scaling. A does not, and should not: A regenerates and contracts
the whole population on every device by definition (`docs/02` C1.3), so only its evaluation
shards while its contraction stays replicated. 0.258 is that mixture, and it is the first
time the A/B distinction has cost what the design says it costs.

**The change is numerically inert.** With `d=512, N=256, D=8`, the update is **bitwise
identical** with and without the constraint, for both strategies, relative difference
`0.000e+00`. It moves work between devices; it does not move a number.

**It costs one all-gather of `4N` bytes**, which is the shaping barrier `docs/02` C1.6
always said was there:

```
how=A  unconstrained -> {'all-gather': 1024}
how=A  constrained   -> {'all-gather': 2048}
how=B  unconstrained -> {'all-reduce': 6291456}
how=B  constrained   -> {'all-gather': 1024, 'all-reduce': 6291456}
```

1024 bytes is `4 x 256` at `N=256`. On B that is 0.016% on top of the existing model-sized
all-reduce. The barrier was previously free because a replicated fitness needs no gathering,
which is not a saving anyone would want.

---

## What was done, and what is still open

**Applied.** The constraint went in `ShardedES.apply`, on the value the returned callable
produces, because that is where the guarantee belongs: `apply` is public API, and someone
scoring a population without calling `tell` should still get a distributed evaluation. It
keeps the strategy protocol sharding-agnostic, since `apply` is a method on `ShardedES`
which already owns the mesh. `tell`'s `replicated` constraint is unchanged and now reads as
what it always was, an explicit gather of `4N` bytes for the sort.

Measured after the change, per-device eval FLOPs from `D=1` to `D=8`:

| | eval | full generation |
|---|---|---|
| `iid_gaussian/A` | 0.125 | 0.258 |
| `iid_gaussian/B` | 0.125 | **0.125** |
| `lowrank_r1/A` | 0.126 | 0.133 |
| `lowrank_r1/B` | 0.126 | 0.126 |

**Also applied: the `x` replication this docstring had described for some time without the
line existing.** It is required by the common-random-numbers argument and it is inert
(verified below), but the "Received incompatible devices" failure the docstring describes
could **not** be reproduced on simulated devices, with or without it. It is restoring a
documented contract, not a demonstrated fix, and that distinction should survive into
whatever this file becomes.

**Verified numerically inert.** Old code against new, same backend, 4 strategies x A/B x
`D` in {1,2,4,8}: **32 of 32 trajectory digests identical**. An earlier attempt compared the
new code on CPU against the committed A100 digests and found 32 of 32 *differing*, including
at `D=1` where the change cannot do anything. That was the backend, not the change, and it
is the same mixed-environment error `check.py` exists to refuse.

Still open:

1. **Whether `tell` should still replicate**, and what `group_relative`'s `(n, g)` fitness
   should do. `P("pop")` is correct at any rank, verified for `(n,)`, `(n, g)` and
   `(n, g, h)`, but the layout deserves its own thought.
2. **`sharding.shard_perturbation` and `sharding.per_member` are now provably dead.**
   They constrain the producer, which is measured to do nothing, and they are called only
   from `tests/test_sharding.py`. Left in place rather than deleted in the same change.
3. **The sweep's numbers do not change retroactively.** `docs/03` records what the library
   did on 2026-08-06. Re-running it would now produce a different scaling curve, and that is
   a new measurement rather than a correction.

`docs/03`'s M1 result and the `1/D` reading stand either way. They describe what the library
did on 2026-08-06, and that does not change retroactively when the cause is fixed.
