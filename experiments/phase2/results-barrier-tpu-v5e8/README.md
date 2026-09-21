# E10: the shaping barrier on TPU v5e-8, 2026-08-19

All 84 cells of `barrier-tpu.yaml` in one session of the `kaggle/t3barrier/` kernel,
pinned at bf477dd. 8x TPU v5 lite, jax 0.11.1, clean worktree. Regenerate the numbers
with `barrier.py --config barrier-tpu.yaml`; the decomposition below is per-iteration
median with the `none` row subtracted where marked.

The E10 answer, in one table (microseconds per generation):

    N        D | gather (none)   centered - none   sort (cr - none)
    64       8 |          5.9              7.1              12.4
    1024     8 |          6.0              6.8              54.2
    16384    8 |          6.0             10.1             720.2
    262144   1 |          3.0              1.2           12099.2
    262144   8 |          5.6             24.3           12135.1

- **The barrier is not a communication problem at any measured scale**, but the gather
  is not what this file first said it was. See the correction below: it costs 4 to 10 us
  at populations up to a thousand and 20 to 26 us at N=262144, where the payload is
  1 MiB and it is bandwidth-bound. docs/02 C1.6 asked what the synchronization costs; on
  this fabric, tens of microseconds at most.
- **The sort is the whole cost at scale, and it is local compute paid redundantly.**
  12.1 ms at N=262144, within 0.3% between D=1 and D=8: every chip sorts the same
  replicated array, exactly as `tell`'s docstring says the design accepts. About
  46 ns per member on this chip, growing slightly faster than linear.
- **Scale anchor:** at the N these papers actually run (30 to a few thousand), the
  whole barrier is under 60 us against generation times in the milliseconds. It only
  becomes a leading term where N reaches EGGROLL territory AND the strategy is cheap;
  at N=262144 a 12 ms sort would rival a low-rank generation itself. A psum-based
  shaping or a sharded sort would remove it, at the cost of shapings declaring their
  communication, the trade `tell` documents declining.

The isolation caveat is now closed in context (`results-barrier-context`,
T6 session, 2026-08-21): full generations with shaping on vs off at the sweep's
largest cells show median deltas of 41 us (d=512, N=1024) and 149 us (d=2048,
N=256), both inside their repeat spreads of roughly +/-100-250 us and within or
below what the isolation numbers allow. At measured scales the barrier is
statistically invisible inside a generation; the 12 ms at N=2^18 stands as the
isolation ceiling for scales where a full generation does not fit this hardware
anyway.


## Correction, 2026-09-21: the `none` row is not the gather

This file read the `none` column as the fitness gather and reported it at 2.4 to 6 us,
flat in N. `barrier_report.py` prints why that is wrong, from these same records:

    what D=8 adds over D=1, microseconds
    N        payload |  none   centered   centered_ranks   ladder all-gather step
    64         256 B |   3.5        9.5             10.2
    256        1 KiB |   3.2       10.0              9.7                      3.8
    1024       4 KiB |   2.9        9.1              8.6                      5.0
    16384     64 KiB |   2.5       12.5             16.6
    262144     1 MiB |   2.6       25.7             38.5                     19.8

    growth from the smallest payload to the largest
    none -0.9   centered 16.2   centered_ranks 28.3   ladder 15.9

A real all-gather is latency-bound when small and bandwidth-bound when large, so what
eight devices add has to grow with a 4096-fold payload. It does for the two programs
that consume what they gather, and by the same 16 us the dedicated ladder measures for
this gather on this chip. It does not for `none`.

The reason is in the program. Under `none` the step replicates `f`, adds a multiple of
it to `f` elementwise, and re-shards the result, so each device's next carry depends
only on its own slice of the gathered array. Nothing downstream needs the other slices,
and the compiler is free to drop the gather or leave it off the critical path. The
chain guards against hoisting and not against this. A mean, a standard deviation or a
rank needs every member, so `centered` and `centered_ranks` cannot skip it. And 1 MiB
in 2.6 us would be about 340 GiB/s, against the 47 GiB/s the ladder fits for this
fabric.

What stands: every sort figure, which is read at D=1 where nothing is communicated
(12.1 ms at N=262144, 46 ns per member, within 0.32% between D=1 and D=8, the 38 us
between them being the gather and repeat noise), the conclusion that the barrier is
local compute rather than communication, and the in-context closure. What is withdrawn:
"2.4 to 6 us, flat in N, about 3 us dearer at D=8". That was the loop's own overhead.
Also narrowed: the whole barrier is 60 us or less up to N of about a thousand, not "a
few thousand"; at N=4096 it is 0.19 ms.

The measurement code is unchanged, so these records still correspond to it. A rerun
that wants the gather from this harness needs the `none` step to consume the replicated
array in a way that depends on every member, as the ladder's chain does.
