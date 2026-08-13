# Making a scan-shaped evaluation distribute: four options, and a recommendation

Written 2026-08-11 for the defect in `diagnosis-seed-regenerated-scan.md`. Per `CLAUDE.md`
working style, both defensible answers are written down with the tradeoff rather than one
being picked silently.

> **Decided 2026-08-11: option B.** `ShardedES.apply` reshapes the member axis to `(D, n/D)`
> and vmaps over it. The `shard_map` implementation is kept in history at `5aecfea` rather
> than in the tree, and the reasoning for rejecting it is summarised in `apply`'s docstring
> so it is not something the next reader has to rediscover by trying it.

## What has to be true

1. `SeedRegenerated` must evaluate `n/D` members per device, not `n`.
2. The three strategies that already distribute must keep distributing.
3. The user's model must keep working. It is arbitrary JAX, and the MuJoCo rollout in
   `tests/test_control.py` is the case that has teeth: it carries its own `lax.scan`.
4. `O(|params|)` per-device storage for `SeedRegenerated` must survive. That is the entire
   bet of the strategy, so a repair that materializes `n` perturbations has repaired nothing.
5. The strategy protocol should stay silent about devices, which is the trade
   `sharding.AXIS_TYPE_NOTE` already declined once.

## Why the obvious repair is not one

Replacing the scan with a `vmap` satisfies 1, 2 and 3 and destroys 4. `seed_regenerated.py`
says so itself: "vmapping the regeneration would materialize all n perturbations again and
buy nothing... If a profile shows an (n, ...) array here, the implementation has quietly
become IIDGaussian with extra steps." It is also what makes `seed_regenerated` sit at 15 MiB
per device where `iid_gaussian/A` needs 12810 (M6). Not a candidate.

## Option A: run the evaluation under `shard_map`

`ShardedES.apply` wraps the whole evaluation in `shard_map`, hands each device its shard of
`member_ids`, and re-derives the perturbation locally. Mirrors `contraction.contract_sharded`
exactly. **Implemented and measured; this is commit `5aecfea`.**

- Satisfies 1, 2, 4, 5. `shard_map` partitions by construction, so it does not care whether
  the strategy's body is a vmap, a scan, or something a user invented.
- **Fails 3.** Three tests in `tests/test_control.py` break with
  `scan body function carry input and carry output must have equal types`, raised inside
  `mujoco_playground/_src/mjx_env.py:174`. The rollout's scan carry derives from the
  replicated batch, so inside the manual mesh it starts invariant and gains varying-ness from
  the loop body, which `lax.scan` rejects. `contract_sharded` documents the same hazard for
  its own accumulator.
- Needs `jax.lax.pcast(params, (POP,), to="varying")` at the boundary or `LowRank.gather`
  fails on the embedding path with a context-mesh mismatch. That part is fixed and tested.
- The unfixed part may well yield to pcasting `x` as well. **Untested.**

The deeper objection is not the three tests. It is that putting the user's model inside a
manual mesh makes the manual mesh part of the library's public contract. Every `lax.scan`,
every sharding annotation and every collective a user writes inside their rollout now has to
be manual-mesh-compatible, and none of that is visible in the API.

## Option B: reshape the member axis and vmap over it

`ShardedES.apply` reshapes `member_ids` from `(n,)` to `(D, n/D)`, vmaps the strategy's
evaluation over the leading axis, constrains that axis to `P("pop")`, and reshapes the result
back to `(n, ...)`. The strategy still receives `n/D` ids and still scans them.

**The point: a vmap batch axis is something GSPMD partitions natively.** The population gets
divided without anything entering manual mode, so the user's model runs in exactly the
context it runs in today.

**Implemented and measured.** All of:

- `test_every_strategy_evaluates_only_its_own_shard`, 15/15 over five strategies at
  `D=2,4,8`.
- `test_the_evaluation_distributes_across_devices`, 6/6, so the vmap strategies still fall
  as `1/D` in FLOPs.
- `tests/test_control.py`, **6/6**, which is where option A fails.
- The whole suite, **732 passed, 0 failed**. Option A is 729 passed, 3 failed.

And it distributes for real, which needed an end-to-end measurement because
`cost_analysis().flops` is blind to a scan's trip count (see the diagnosis). On 8 simulated
CPU devices sharing cores, `seed_regenerated` at `n=64, d_model=64`, median of 5:

| | `D=1` | `D=8` |
|---|---|---|
| pre-fix | 39.6 ms | **50.5 ms**, slower |
| option B | 31.2 ms | **8.2 ms**, 3.8x faster |

Pre-fix, adding devices made it slower: eight devices each evaluating all 64 members. Under
option B the total work falls, which on shared cores is the signal that the work was actually
divided rather than replicated.

Costs and open questions:

- `ShardedES.apply` becomes device-count aware in a new way: it reshapes by `D`. That is
  already true of `member_ids` and `check_population`, so it is not a new kind of knowledge,
  but it is a second place that knows `n` divides `D`.
- The vmap-based strategies gain an outer vmap of extent `D` wrapping their existing inner
  vmap. Measured harmless here, but it is a change to programs that were already correct, and
  it deserves a look at the `d=2048` shapes on real hardware before it is trusted.
- It relies on GSPMD partitioning the constrained leading axis. That is the same propagation
  mechanism whose unstated preconditions caused this defect. The difference is that a batch
  axis is the case GSPMD is built for, rather than the case it silently declines, but it is
  still inference rather than construction.
- Reshape assumes row-major contiguity matches how the mesh splits `(n,)`. It does, and
  `check_population` already refuses uneven splits, but the test suite should say so.

## Option C: a sharding seam in the protocol

Add an optional declaration to the strategy protocol, so `ShardedES` uses `shard_map` only
for strategies that ask for it and leaves the vmap strategies on today's path.

- Satisfies everything, and confines the manual-mesh constraint to the strategies that need
  it.
- Costs the thing `AXIS_TYPE_NOTE` spent its argument protecting: the protocol stops being
  silent about devices. It says to revisit "if the strategy protocol ever grows a
  sharding-aware seam for another reason", so this is at least an honest trigger.
- **The failure mode is the one we are trying to kill.** A user-defined scan-shaped strategy
  that does not set the flag silently does not distribute, which is exactly the bug, now with
  a declaration to forget. The new test catches it for strategies in this repo and cannot
  catch it for anyone else's.

## Option D: chunk the scan

Change `SeedRegenerated` to scan over chunks and vmap within a chunk. The vmap axis is
partitionable, and peak memory becomes `O(chunk * |params| / D)` rather than `O(|params|)`.

- Keeps everything in Auto and needs no change to `ShardedES`.
- Weakens requirement 4 by a factor of the chunk size, and introduces a knob whose right
  value depends on the model, the population and the device count.
- It is a change to a strategy's numerics-adjacent structure rather than to placement, which
  puts it furthest into ground-rule-1 territory.

## Recommendation

**Option B.** It is the only one measured to satisfy all five requirements, it needs no
protocol change, and it keeps the user's model in the execution context it already targets.
Option A is a real solution to the wrong problem: it fixes distribution by moving the user
into a manual mesh, and the three MuJoCo failures are that bill arriving.

The honest caveat is that B leans on GSPMD propagation, and propagation is what failed here.
The mitigation is not faith, it is the test: `test_every_strategy_evaluates_only_its_own_shard`
asserts the property directly, over every strategy, and fails 15/15 on the pre-fix code. What
made this defect expensive was that the existing test could not see it. That is fixed
independently of which option ships.

If B is taken, A's commit should be reverted rather than kept alongside, and the `pcast`
lesson it produced belongs in a comment wherever the mesh boundary ends up.

## What is still not measured

- Neither option has run on real hardware. The timings above are simulated CPU devices,
  which model work and not interconnect.
- `Mirrored(SeedRegenerated())` is covered by the shard test but has never appeared in a
  sweep, so its scaling is unmeasured.
- The M1, M2, M3 and M6 numbers in `docs/03` were measured with `seed_regenerated` not
  distributing. Whichever option ships, its rows change and the sweep should be re-run for
  that strategy. The other three strategies are unaffected: their evaluation already
  distributed and neither option changes what they compute.

---

## Addendum, 2026-08-13: the reshape costs 32% on the materializing strategies

Found by asking whether any Phase 2 measurement is still valid for the code that ships.
Measured, not estimated, and it is a cost this proposal did not price.

**The reshape path re-derives the perturbation inside the vmap, and that is a third
materialization.** Counting noise primitives in the jaxpr of one full generation at
`d_model=64, n=32`:

| | `random_bits` | `normal` |
|---|---|---|
| `a496345`, output constraint | 2 | 32 |
| current, reshape and vmap | **3** | **48** |

`ask` materializes, `apply` re-derives, `contract` re-derives. Before the reshape, `apply`
used the perturbation `ask` had already built, so there were two. **`ask`'s copy is dead and
XLA does not eliminate it**, which is the specific claim `core.apply`'s docstring makes and
which is wrong.

Per-device FLOPs for one generation, against the sweep commit:

| strategy | FLOPs | peak temp memory |
|---|---|---|
| `iid_gaussian` | **1.32x** | 1.18x |
| `lowrank_r1` | 1.08x | 1.12x |
| `mirrored_lr1` | 1.07x | 1.10x |
| `seed_regenerated` | 1.00x | n/a |

The cost lands exactly where the reshape is not needed. `SeedRegenerated.sample` does no
work, so re-deriving it is free; `IIDGaussian.sample` *is* the work, and it is a strategy
whose evaluation a sharding constraint already distributed.

Numerics are unaffected: every trajectory digest matches across the change, and the
collectives are identical.

### Three ways out, and why none of them is obviously right

**(a) Leave it.** Correct by construction: `ShardedES.apply` cannot be fooled by a strategy
whose body it does not understand, which is the property that took two rented sweeps to buy.
Costs 32% of a generation on `iid_gaussian` and roughly 8% on the low-rank strategies.

**(b) Let a strategy declare `evaluation = "batched"` and take the old path.** Implemented
and reverted on 2026-08-13. It removes the cost exactly, restoring `a496345` FLOPs to the
digit, and it reintroduces the failure mode: **a scan strategy that declares itself batched
does not distribute, and nothing can tell.** Measured, a correct `SeedRegenerated` and one
that wrongly declares `batched` both report a FLOP ratio of 1.000, because `cost_analysis`
counts a `while` body once. The declaration would be unverifiable, which is what the
structural version exists to avoid. It also splits the test coverage: the `n/D` ids property
is only true on the reshape path, so nine tests fail and would have to be made path-aware.

**(c) Reuse the perturbation instead of re-deriving it.** Needs the pytree reshaped from
`(n, ...)` to `(D, n/D, ...)`, and only the strategy knows which of its leaves carry a member
axis. A generic "leading dimension equals `n`" rule is wrong for `Mirrored`, whose inner
perturbation has `n/2`, and would silently mis-shape a parameter leaf that happens to be `n`
long. Done properly it is a protocol method, which `sharding.AXIS_TYPE_NOTE` says to add only
with a reason. A third of a generation is arguably a reason.

**Left for Andres.** (a) is what ships today and is the safe default. (c) is the one that
gets both properties, at the cost of the protocol seam the design has so far avoided.

### What this means for the Phase 2 numbers

The `iid_gaussian`, `lowrank_r1` and `mirrored_lr1` rows of M1, M2, M3 and M6 were measured
at `a496345`. Their **scaling ratios** survive the change (`D1->D8` FLOP ratio 0.1260 against
0.1262 for `iid_gaussian`), so parallel efficiency and weak throughput are approximately
unaffected. Their **absolute** ms/generation and MiB/device figures describe a program doing
up to a third less work than the one that ships. `docs/03` should say so until they are
re-run at a single commit, which is 192 configurations and about $30.
