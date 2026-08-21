# E10 in context: shaping on vs off inside full generations, 2026-08-21

One T6 session (`kaggle/t6ctx/`, pinned 59e85ad), one v5e-8 host, D=8. Full
lowrank_r1/B generations at the sweep's largest cells, shaping none vs
centered_ranks, 3 warmup + 7 repeats, from `barrier_context.py`.

    d=512  N=1024: none 2.17 ms, ranks 2.22 ms; median delta 41 us (1.9%)
    d=2048 N=256:  none 5.35 ms, ranks 5.50 ms; median delta 149 us (2.8%)

Both deltas sit inside their repeat spreads (the d=2048 ranks repeats are
bimodal, ~5.31 vs ~5.7, plain TPU jitter) and within or below the isolation
ceiling from `results-barrier-tpu-v5e8` (gather + sort of order tens of us at
these N). The sentence this buys the paper: at measured scales the shaping
barrier is statistically invisible inside a generation, consistent with the
isolation numbers; the isolation 12 ms at N=2^18 remains the ceiling for
scales where a full generation does not fit this hardware.
