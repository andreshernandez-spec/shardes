# T7 contraction isolation: the term between the byte count and the wall clock

`allreduce_ladder.py` times the collectives, `run.py` times whole generations, and the
contraction sits between them: the arithmetic strategy B splits `D` ways and strategy A
repeats on every device. Until it is measured the crossover has two endpoints and no cost
model, which is what the paper's limitation said.

`contraction_isolation.py` measures it at the F2 grid's own cells (`d` in {512, 2048},
`N` in {256, 1024} and {128, 256}, five strategies, matmul precision `highest`, D = all
local devices), with the ladder's discipline: three variants, each timed as a dependent
chain of 1 and 9 contractions, per-contraction cost from the slope so the dispatch floor
(0.32 ms on the A100, 0.60 ms on the v5e) drops out.

| column | what |
|---|---|
| `contraction_seconds` | `C`, the replicated contraction: strategy A's work, plus its 4N gather (~8 us) |
| `contraction_local_seconds` | B's local `N/D` contraction, psum removed |
| `allreduce_insitu_seconds` | B with the psum minus B without it |
| `shard_ratio` | `C / (D * C_local)`; 1.0 means the contraction shards perfectly |

`allreduce_insitu` is the one the paper needs. The ladder times the same psum in an
otherwise empty program; this times it inside a program holding a model's worth of live
buffers. `timemodel.py` reads both and checks

    t_A - t_B = C (D - 1) / D + ag(4N) - ar(4P)

cell by cell against the committed sweeps.

## Why it exists: the open term

Run backwards on the committed grids (`python timemodel.py`, no records here needed), the
model solves four of twenty A100 cells and four of sixteen v5e cells to a NEGATIVE
contraction, which is not a thing. Worst is `mirrored_lr1` d=2048 N=128: B costs 0.71 ms
(A100) and 0.63 ms (v5e) more than its local contraction plus the ladder's all-reduce can
account for. Every one of those cells is low-rank at the largest model size, which is
exactly where the crossover's A-favored side lives, so the deficit is not a corner case.

Three candidates, and this experiment separates them:

1. the in-situ all-reduce is simply dearer than the isolated one (`allreduce_insitu` >
   ladder `step`), from memory pressure or a different XLA collective algorithm at that
   point in the program;
2. B's local contraction does not actually shard `D` ways at these shapes
   (`shard_ratio` well under 1), so B saves less compute than the model credits it with;
3. something outside the contraction differs between the placements, which would show as
   both of the above landing on their predicted values and the residual surviving.

## Status

Pending hardware on both platforms. Plumbing smoke-tested on 8 simulated CPU devices for
all five strategies; those runs are not kept, because 8 processes on one CPU measure the
host, not a fabric. The A100 cells ride the E18b cluster preflight (docs/10) and the v5e
cells ride a Kaggle TPU session with the pending `results-regen` re-measurement.
