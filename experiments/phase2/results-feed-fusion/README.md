# The feed dependence is not the residual either. RTX 3080 Laptop, 2026-09-03

`feed_fusion.py` at 0ddba21, one RTX 3080 Laptop GPU (16 GB), jax 0.11.0, matmul
precision highest, 5 warm-ups discarded and 51 timed repeats. Diagnosis only; nothing
here is cited as a result.

## What was left to test

One cell of the F2 grid keeps a residual the cost model does not explain:
`iid_gaussian` at d=512, N=256, where A is 1.5 ms slower in a real generation than
`contraction_isolation.py` says the contraction costs. It reproduces on two 8x A100
hosts (+1.487 and +1.560 ms). Two explanations were on the table and the first is
already dead: a resident population-sized buffer moves the contraction by under 0.31%
(`../results-recheck-2026-09-03/`).

The second is the one the isolation README raised first. The chain feeds the contraction
from a scan carry; a real generation feeds it from the evaluation. If XLA fuses or
schedules those differently, the chain would understate what a generation pays.

## Why one GPU is the right instrument here

Under placement A every device contracts the whole population, so A's local contraction
is the same computation at D=1 as at D=8; only B's shrinks with D. The residual is an
A-side residual, so a single device measures the right thing. The limit is that a single
device cannot see anything that needs the collective.

Two programs, differing in one edge and nothing else:

| | ask | apply | tell's weights |
|---|---|---|---|
| `tell_fed` | yes | yes | apply's output |
| `tell_cut` | yes | yes | an argument, with apply's output summed into a second output so it is still computed |

Both pay `ask` and both pay `apply`, so neither cancels and neither can be dropped as
dead code. `tell_fed - tell_cut` is the dependence, priced.

## Result: the dependence is free, and if anything negative

| run | tell_fed ms | tell_cut ms | gap ms | % of generation |
|---|---|---|---|---|
| 1 | 39.99 | 40.89 | -0.90 | -2.19 |
| 2 | - | - | -1.18 | -2.88 |
| 3 | - | - | -1.27 | -3.09 |
| 4 (committed record) | 39.95 | 40.96 | -1.02 | -2.49 |

Feeding the contraction from the evaluation is consistently about 1 ms *faster*, not
slower. The hypothesis predicts slower.

**The measurement is sensitive enough for this to mean something.** On the A100 the
residual is 1.5 ms against an 8.9 ms A-side generation, 17.5% of it. This machine runs
the same generation in 40 ms, so a mechanism that scaled with the work would show as
about 7 ms here. The gap is -1.0 ms with a run-to-run range of 0.4 ms. The effect is
absent with a wide margin, not merely lost in noise.

## What that leaves

Both named explanations for the surviving residual are now dead: memory pressure on the
A100 at D=8, and the feed dependence at D=1 here. The cell is still 1.5 ms slower under
A than the decomposition accounts for, on two hosts, and nothing on the list explains it.

What this cannot rule out is anything that needs more than one device: interaction
between the replicated contraction and the fitness all-gather, or scheduling around the
collective. Testing that needs 8 devices and a new hypothesis, and there is no candidate
worth renting for yet.

N=1024 was not measured: its population is 6 GB materialized and the card has 16 GB, so
it exceeded memory during compilation. The residual on that cell did not reproduce on
the A100 either (`../results-recheck-2026-09-03/`), so there is nothing to explain there.
