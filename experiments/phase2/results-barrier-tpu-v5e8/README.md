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

- **The barrier is not a communication problem at any measured scale.** The gather is
  2.4 to 6 us: flat in N (the payload at N=262144 is 1 MB) and about 3 us dearer at
  D=8 than D=1. docs/02 C1.6 asked what the synchronization costs; on this fabric,
  single-digit microseconds.
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
