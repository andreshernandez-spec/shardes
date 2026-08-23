# E19: the positive control for the E13 null. A100, 2026-08-22

Review 7 objected that the E13 null (no held-out difference across ranks at N=30) had no
arm expected to be modestly worse, so it could not be told from a task that resolves
nothing. E19 runs that arm: the full-rank and rank-1 recipes of E13 at population 16
instead of 30, 940 generations instead of 500, so the total training samples match
(940 x 16 x 8 = 120,320 against 120,000). The square-root law (Section 7) predicts
0.73x the alignment per update at N=16; the prediction was frozen in the config headers
(`e19-n16-seed.yaml`, `e19-n16-lr1.yaml`) before the runs. Code at 792a1fd (eval chunk
4, since 16 members on one device do not divide by 5), jax 0.11.1, one community
A100-SXM4-80GB, seeds 0-2, run back to back on one pod. `plot_e19.py` draws F7c and
prints the numbers below.

Final held-out eval, mean [min-max] over seeds, beside the E13 arms at N=30 (clean rerun):

| arm | N=16 (E19) | N=30 (E13) |
|---|---|---|
| full rank | 0.157 [0.154-0.162] | 0.156 [0.151-0.162] |
| rank 1 | 0.154 [0.152-0.157] | 0.157 [0.154-0.160] |

First evaluation whose seed-mean is at or above 0.15: 24,064 samples for both N=16 arms,
36,000 and 48,000 for the N=30 arms (evaluations fall every 94 updates at N=16 and every
50 at N=30, so the grids differ; the N=16 curves are not behind at any common point of
F7c). The control did not separate: a population that the law says should align 27%
worse per update reaches the same plateau on the same sample budget, inside seed noise.

What that means for E13. The task saturates at about 0.155 held-out reward once the
format is learned (every arm, every population, by 24,000-36,000 samples), and the
plateau is not moved by the population, the rank, or freezing 27% of the parameters. So
the E13 null is a property of the task ceiling at least as much as of the ranks: it
bounds nothing finer than a 27% alignment change, which it also does not resolve. The
frozen prediction (0.73x alignment, visible as a slower or lower curve) was wrong about
the curve, not about the alignment, which E19 did not measure.

Timing, steady-state updates 2-939, median per seed: full rank 2.39 / 2.40 / 2.38 s,
rank 1 2.49 / 2.57 / 2.47 s. At N=16 rank 1 is not cheaper per update. The E13
per-update ratio (33% less at N=30) is a population-dependent number: the cost surfaces
of Section 6 put the regeneration cost rank 1 removes in proportion to N, and at 16
members it no longer covers whatever rank 1 adds. Which part of the update that is was
not profiled here.

Compilation-scale generations 0 and 1: 325-564 s. Cost: one pod, 20:19 to 02:50 UTC
including the jinja2 relaunch (see `../e13-a100-2026-08-22-clean/README.md`) and the
first attempt that died on the eval chunk, 6.5 pod-hours, $9.
