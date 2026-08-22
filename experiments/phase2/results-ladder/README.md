# T7 collective ladder: what the two placements' transfers cost in isolation

Review 7 asked for cost attribution: the contraction-placement results report bytes
and end-to-end seconds, never the isolated time of the transfer each placement issues.
`allreduce_ladder.py` times the two collectives on the full local mesh with the sweeps'
discipline (3 warm-up, 10 fenced repeats, median and IQR): strategy B's model-sized
all-reduce, issued as `contraction.contract_sharded` issues it (every device holds a
full-size partial, `shard_map` + `psum`), and the 4N-byte fitness all-gather both
placements share. One JSON per platform, stamped with the environment.

Two numbers per size. `call` is a one-op program timed from Python, dispatch and sync
included: the floor a collective pays when it is the whole program. `step` is the cost of
one more collective inside a program that already runs, from the slope of a dependent
chain of 1 and 9 collectives minus the slope of the same chain without the collective.
Each chain step adds a per-device term so XLA cannot prove the operand replicated,
which would let its all-reduce simplifier replace the collective with a multiply. `step` is the number the ES loop pays: the
psum sits inside the generation program.

## 8x A100-SXM4-80GB (RunPod SECURE, NVLink), D=8

`ladder-nvidia-a100-sxm4-80gb-D8.json`, commit 41b04b9, clean worktree, jax 0.11.1.

| payload | what | call (us) | step (us) |
|---|---|---|---|
| 8 B | latency point | 323 | 27 |
| 1 KiB | | 359 | 32 |
| 1 MiB | | 376 | 51 |
| 6 MiB | d=512 block, 1.57M float32 | 497 | 117 |
| 96 MiB | d=2048 block, 25.2M float32 | 1426 | 925 |
| 100 MiB | bandwidth point | 1513 | 1024 |
| all-gather 1 KiB | N=256 fitnesses | 290 | 8 |
| all-gather 4 KiB | N=1024 | 284 | 8 |
| all-gather 1 MiB | N=2^18 | 312 | 21 |

Fit: alpha 27 us, beta 98 GiB/s of payload (the 8 B and 100 MiB step points).
The one-op floor is 0.32 ms, twelve times the in-program latency, which is why the
`call` column is not the number to read against a generation.

Read against the cost surface (`results-cost/`, same node class, D=1 per-device
times): the d=2048 block spends about 150 ms per generation at N=256 in bfloat16, so
B's 0.9 ms all-reduce is under 1% of it, and A's two 4N gathers at N=1024 are 8 us
each. At d=512 the 0.12 ms all-reduce sits against a 12 ms generation. On this fabric
neither placement's transfer is where the A/B gap comes from; the gap is the contraction
compute, split D ways under B and replicated under A.

## TPU v5e-8 (Kaggle), D=8

Pending: the first v5e session (kernel `kaggle/t7ladder`, commit 8d06d64) reduced a
(D, n/D) array, so every labelled payload was an eighth of what B moves, and its
numbers are not kept. The corrected script runs at the head of the `kaggle/e17btpu`
kernel; its JSON and table land here when that session returns. For orientation only,
the mislabelled run put the one-op floor at 0.61 ms and the 12.5 MiB reduction at
0.95 ms.
