# T7 contraction isolation: the term between the byte count and the wall clock

`allreduce_ladder.py` times the collectives, `run.py` times whole generations, and the
contraction sits between them: the arithmetic strategy B splits `D` ways and strategy A
repeats on every device. Until it was measured the crossover had two endpoints and no cost
model.

`contraction_isolation.py` measures it at the F2 grid's own cells (`d` in {512, 2048},
`N` in {256, 1024} and {128, 256}, five strategies, matmul precision `highest`,
D = all local devices), with the ladder's discipline: three variants, each timed as a
dependent chain of 1 and 9 contractions, per-contraction cost from the slope so the
dispatch floor (0.32 ms on the A100) drops out.

| column | what |
|---|---|
| `contraction_seconds` | `C`, the replicated contraction: strategy A's work, plus its 4N gather (~8 us) |
| `contraction_local_seconds` | B's local `N/D` contraction, psum removed |
| `allreduce_insitu_seconds` | B with the psum minus B without it |
| `shard_ratio` | `C / (D * C_local)`; 1.0 means the contraction shards perfectly |

## Results: 8x A100-SXM4-80GB, D=8, 2026-08-25

All 20 cells, no failures, 12 minutes of one node. Commit `c293481`, jax
0.11.1, `matmul_precision=highest`.

| strategy | d | N | C ms | C/D ms | ar in situ us | shard |
|---|---|---|---|---|---|---|
| iid_gaussian | 512 | 256 | 3.933 | 0.818 | 176 | 0.60 |
| iid_gaussian | 512 | 1024 | 14.171 | 2.496 | 190 | 0.71 |
| iid_gaussian | 2048 | 128 | 31.915 | 4.111 | 1156 | 0.97 |
| iid_gaussian | 2048 | 256 | 60.894 | 9.478 | 1301 | 0.80 |
| lowrank_r1 | 512 | 256 | 0.214 | 0.105 | 169 | 0.26 |
| lowrank_r1 | 512 | 1024 | 0.509 | 0.140 | 175 | 0.46 |
| lowrank_r1 | 2048 | 128 | 0.778 | 0.356 | 1637 | 0.27 |
| lowrank_r1 | 2048 | 256 | 1.264 | 0.368 | 1669 | 0.43 |
| mirrored_lr1 | 512 | 256 | 0.153 | 0.091 | 175 | 0.21 |
| mirrored_lr1 | 512 | 1024 | 0.355 | 0.125 | 173 | 0.35 |
| mirrored_lr1 | 2048 | 128 | 0.495 | 0.339 | 1637 | 0.18 |
| mirrored_lr1 | 2048 | 256 | 0.781 | 0.346 | 1623 | 0.28 |
| mirrored_seed | 512 | 256 | 3.649 | 0.633 | 174 | 0.72 |
| mirrored_seed | 512 | 1024 | 14.124 | 2.307 | 206 | 0.77 |
| mirrored_seed | 2048 | 128 | 15.148 | 2.786 | 1066 | 0.68 |
| mirrored_seed | 2048 | 256 | 30.727 | 4.121 | 1261 | 0.93 |
| seed_regenerated | 512 | 256 | 7.065 | 1.188 | 179 | 0.74 |
| seed_regenerated | 512 | 1024 | 29.052 | 3.587 | 164 | 1.01 |
| seed_regenerated | 2048 | 128 | 30.722 | 4.111 | 1264 | 0.93 |
| seed_regenerated | 2048 | 256 | 61.057 | 7.781 | 1249 | 0.98 |

## What it settles

The paper's open term was eight of eleven A-favored cells where B was slower than its
local contraction plus the ladder's all-reduce could account for, which a contraction
cannot be. Two candidates were on the table. **Both are real, and both are worst exactly
where the crossover lives.**

**1. The contraction does not shard at 1/D.** `shard_ratio` is 0.18-0.46 across the
low-rank family and 0.60-1.01 across the dense and seed-regenerated arms. The split is
the size of the contraction, not the strategy: where `C` is 0.15-1.3 ms the per-device
fixed cost dominates and B keeps a fifth to a half of its ideal saving; where `C` is
3.6-61 ms it amortises and B gets nearly all of it (`seed_regenerated` d=512 N=1024
reaches 1.01). So `C (D-1)/D` overstates what B saves, most on the low-rank arms.

**2. The same collective costs more in a leaner program.** The 96 MiB psum costs
1.75-1.80x the ladder's isolated step inside the low-rank programs and 1.15-1.41x inside
the dense and seed ones. A payload does not have a cost; it has a cost in a program, and
the lean low-rank generation has less compute to hide the transfer behind. The ladder's
number is a floor, not an estimate.

Replacing `C (D-1)/D` with the measured `C - C_local`, and the ladder's step with
`allreduce_insitu`, closes the four low-rank d=2048 cells that drove the open term:

| cell | residual, ladder + C(D-1)/D | residual, measured |
|---|---|---|
| lowrank_r1 d=2048 N=128 | -0.983 ms | **-0.012 ms** |
| lowrank_r1 d=2048 N=256 | -1.031 ms | **-0.077 ms** |
| mirrored_lr1 d=2048 N=128 | -1.148 ms | **-0.159 ms** |
| mirrored_lr1 d=2048 N=256 | -1.170 ms | **-0.224 ms** |

Across all 20 cells the median absolute residual falls from 1.125 ms to 0.626 ms and sign
agreement rises from 17/20 to 18/20. Neither mechanism closes the low-rank cells alone;
both are needed.

## What it does not settle

**Most of the dense-side residual is the sweep's own repeat noise.** Compared against
the spread of each cell's five timed repeats, 18 of 20 cells agree within twice that
spread. The largest absolute residual, 9.2 ms at `iid_gaussian` d=2048 N=128, sits on a
cell whose own repeats span 10.5 ms: there is nothing there to explain. The dense and
seed arms have long generations and wide spreads (up to 14 ms), so a residual of a few ms
on them is not evidence of a missing term.

**Two cells do exceed it**, both `iid_gaussian` at d=512: N=256 at 1.487 ms against a
0.588 ms spread (2.5x) and N=1024 at 5.392 ms against 0.702 ms (7.7x). Both are positive,
meaning A is slower in a real generation than the isolated contraction accounts for.
`iid_gaussian` is the only arm that materializes the population, so the leading candidate
is that A's replicated contraction runs while `ask`'s materialized population is still
resident and competes for memory bandwidth, which the isolation harness cannot see
because nothing else is resident there. `contraction_isolation.py --resident` would test
it by allocating a population-sized buffer before timing. Two cells is a thin basis for a
rental; the question is real but small.

**The two sign misses are both near-parity cells** (`lowrank_r1` d=512 N=256 at +0.017 ms
measured, `mirrored_lr1` d=512 N=1024 at -0.075 ms), where the model and the measurement
disagree by less than either differs from zero.

**The residual analysis above is A100 only.** The v5e half was measured afterwards and
is below; the repeat-noise comparison and the two surviving cells have not been redone
against it. The two fabrics differ by 2.17x on the isolated 96 MiB all-reduce, so the
absolute spreads do not transfer even where the mechanisms do.

## A correction

An earlier version of this file, written from the six E18 preflight cells, said that a
positive measured `C` "kills one of the two candidates ... the shortfall is not a
contraction that fails to shard; it is B paying more for its all-reduce." That was wrong.
A positive `C` shows only that B saves something, not that it saves `C (D-1)/D`, and the
`shard_ratio` column it could not see is 0.18-0.46 on exactly those cells. Both mechanisms
were needed and one of them had been argued away on insufficient evidence.

## The v5e half (2026-08-31)

Measured on a Kaggle TPU v5e-8 by the `kaggle/e17btpu` session that also resumed the
E17b grid, pinned at 1ba0dd0, matmul precision `highest`, D=8, all 20 cells. The
`--resident` flag exists at that commit but was not passed, so the measurement path is
the one the A100 cells used; the records carry `"resident": false` where the A100 ones
predate the field.

| what | A100 (NVLink) | v5e (ICI) |
|---|---|---|
| `C/(D C_local)`, dense and seed arms | 0.60-1.01 | 0.95-1.00 |
| `C/(D C_local)`, low-rank arms | 0.18-0.46 | 0.16-0.62 |
| in-situ all-reduce over the ladder's step | 1.49-1.77x | 1.18-1.60x |
| forward model, cells with the right sign | 18 of 20 | 15 of 16 |
| cells whose backwards solve wants a negative C | 4 | 4 |

Both mechanisms are present on both fabrics and at the same sizes, which is what makes
them properties of the placement rather than of one machine: the contraction shards
nearly perfectly when it is large (dense, tens of ms) and poorly when it is small
(low-rank, tenths of a ms), and the same collective costs more inside a lean program
than the ladder measures it at alone. The two fabrics differ by 2.17x on the isolated
96 MiB all-reduce and still fail in the same two ways.

`timemodel.py` prints both tables; the paper quotes them in prose rather than typesetting
them (`tb7.tex` is the E18 host-boundary table, a different result).
