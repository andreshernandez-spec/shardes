# The dense-side residual, rechecked. 8x A100-SXM4-80GB, 2026-09-03

One SECURE 8x A100-SXM4-80GB node (NV12 all-to-all NVLink), code at 158a930, jax 0.11.1,
`recheck.sh` in this directory, about 25 minutes of uptime, $5.30. `recheck_report.py`
prints every number below. Three questions, three answers, two of them negative.

Contents: `run1`..`run5` are five separate invocations of `sweep-iid512-recheck.yaml`
exactly as committed, each into a fresh directory. `contraction-plain` and
`contraction-resident` are `contraction_isolation.py` with and without `--resident`, both
measured here so the pair is matched on one host.

## 1. Between runs, the sweep is far more reproducible than assumed

Five fresh processes, five compilations, same config:

| cell | placement | spread ms | sd ms |
|---|---|---|---|
| d=512 N=256 | A | 0.093 | 0.038 |
| d=512 N=256 | B | 0.110 | 0.044 |
| d=512 N=1024 | A | 0.155 | 0.059 |
| d=512 N=1024 | B | 0.059 | 0.024 |

This was the worry that motivated the session: the sweep's five repeats are consecutive
generations inside one process after one compile, so `residual_spread.py` bootstraps a
floor rather than the real uncertainty. It turns out to be a tight floor. Between-run sd
is 0.02-0.06 ms, at or below the within-run bootstrap's 0.09-0.16 ms for the same cells,
so compilation and process variation add nothing on top of steady-state jitter here. The
caveat stands as written but is empirically small.

Note what that does to `sweep-iid512-recheck.yaml`'s premise. It raised the repeats from
5 to 25 in case five was too thin. Five was not the problem.

## 2. One residual reproduces, one does not

Model evaluated against this session's own isolation records:

| cell | measured A-B ms | predicted ms | residual ms | on the original host |
|---|---|---|---|---|
| d=512 N=256 | +4.326 | +2.767 | **+1.560** | +1.487 |
| d=512 N=1024 | +11.869 | +12.479 | **-0.611** | +5.392 |

**N=256 reproduces**, +1.56 against +1.49 ms on a different machine. It is a real property
of the placement at that shape and the model does not account for it.

**N=1024 does not.** It was the larger of the two, at 33x its within-run noise, and here
it is gone and slightly negative. The cell itself moved: A ran 31.0 ms on the original
host and 25.1 ms here, while the isolated contraction moved with it, and the shard ratio
went from the 0.18-0.46 band the low-rank cells sit in to 0.863. So the residual was a
property of that host or that session, not of the placement, and the model describes this
cell to within 5%.

That halves the open question. One cell asks it, not two, and the one that survives is the
smaller one.

## 3. A resident population-sized buffer does not slow the contraction

The leading explanation for those residuals was memory pressure: `iid_gaussian` is the only
arm that materializes its population, so A's replicated contraction may run while that
memory is still resident, which the isolation harness cannot see because nothing else is
resident there. `--resident` allocates a population-sized buffer before timing.

| cell | C plain ms | C resident ms | change | resident MiB/dev |
|---|---|---|---|---|
| d=512 N=256 | 3.821 | 3.809 | -0.31% | 192 |
| d=512 N=1024 | 14.654 | 14.641 | -0.09% | 768 |
| d=2048 N=128 | 32.064 | 32.075 | +0.03% | 1536 |
| d=2048 N=256 | 61.035 | 61.025 | -0.02% | 3072 |

**Nothing, at any size.** Every change is under 0.31% and under 13 microseconds on
contractions of 3.8 to 61 ms, with up to 3 GiB per device resident, and the signs are
mixed, which is what noise looks like. The hypothesis is refuted, not merely unsupported:
if occupancy were worth 1.5 ms at d=512 N=256 it would have shown here, and the largest
buffer tested is sixteen times that cell's.

## Where that leaves the residual

One cell, `iid_gaussian` d=512 N=256, is 1.5 ms slower under A than the decomposition
accounts for, reproducibly, on two hosts, and the memory-pressure explanation is dead.
What remains is the candidate the isolation README raised first and this session does not
test: the chain measures a contraction fed by a carry while a real generation feeds it
from `apply`, and XLA may fuse the two differently. That is a compiled-HLO question, not a
rental question, and it can be answered on any machine with a GPU.

Nothing here changes a headline. The placement crossover, its signs, and the two
mechanisms behind the open term are all unaffected; this only sharpens what is left
unexplained, from "two cells and a hypothesis" to "one cell and a different hypothesis".
