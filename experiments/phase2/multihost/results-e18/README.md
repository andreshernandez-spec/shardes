# E18: the contraction crossover across a real host boundary. A100, 2026-08-24

Two nodes of 8x A100-SXM4-80GB on a RunPod Instant Cluster (US-MD-1), commit
`0826d5d`, 43 min of cluster time. The design and frozen hypotheses are
docs/10; predictions were frozen by `predict.py` (from the measured C plus the
committed D=8 sweep) before any 2x8 campaign cell, enforced by the driver order.

`analysis` reproduces the table below from the cell JSONs in this directory.
36 timed cells: 2 arms x 2 hows (A, B) x 3 (N, d) x 3 topologies (1x8, 2x4, 2x8),
warmup 3, repeats 7, matmul precision highest.

## Fabric: socket, not InfiniBand

The cluster's InfiniBand is present (mlx5 HCAs) but NCCL fails it with
`IBV_WC_RETRY_EXC_ERR` on every host we drew (5 clusters). The launcher probes
IB, walks the known RoCE/GID/HCA fixes, and, when none work, falls back to the
socket transport over the overlay. All numbers here are over that socket
transport and are stated relative to its measured bandwidth. Measured by the
preflight psum ladder:

| link | alpha | beta |
|---|---|---|
| NVLink, intra-node (1x8) | 326 us | 515 GiB/s |
| socket, inter-node (2x8) | 430 us | 9.2 GiB/s |

A ~56x bandwidth drop at the host boundary, one to two orders as docs/10 framed
it, though the top end is socket-over-Ethernet, not the IB the design wanted.

## The measurement (t_B - t_A, ms; negative = B faster, positive = A faster)

| arm | d | N | 1x8 NVLink | 2x4 boundary D=8 | 2x8 boundary D=16 | predicted 2x8 |
|---|---|---|---|---|---|---|
| seed_regenerated | 2048 | 128 | -21.4 (B) | +86.8 (A) | +74.1 (A) | B (miss) |
| seed_regenerated | 2048 | 256 | -50.8 (B) | +57.2 (A) | +56.7 (A) | B (miss) |
| seed_regenerated | 512 | 1024 | -20.1 (B) | -13.0 (B) | -24.8 (B) | B (hit) |
| mirrored_lr1 | 2048 | 128 | +1.6 (A) | +108.9 (A) | +159.1 (A) | A (hit) |
| mirrored_lr1 | 2048 | 256 | +1.4 (A) | +111.5 (A) | +130.9 (A) | A (hit) |
| mirrored_lr1 | 512 | 1024 | ~0 (tie) | +6.9 (A) | +7.8 (A) | A (hit) |

## Findings

**H1, the boundary cliff, is large and clean.** 1x8 vs 2x4 is the controlled
test: same 8 devices, boundary off then on. For `seed_regenerated d=2048 N=128`
`t_B - t_A` swings from -21 ms to +87 ms, so turning the boundary on costs
strategy B about 108 ms. B pays the model-sized all-reduce every generation and
that all-reduce goes from NVLink to a 9.2 GiB/s socket. The `mirrored_lr1 d=2048`
cells show the same shape (+1 ms to +109 ms).

**A real fabric-driven crossover (H2/H3).** `seed_regenerated d=2048` wins as B
on NVLink and flips to A across the boundary, at both N. The decision rule
inverts at the host boundary for the expensive-perturbation arm. The small model
`seed_regenerated d=512` keeps B everywhere (its all-reduce is cheap), and
`mirrored_lr1` stays A throughout with the margin growing at the boundary.

**H4: the model gets 4 of 6 signs, and the two misses are the finding.** Both
misses are `seed_regenerated d=2048`: `predict.py` predicted B still wins at 2x8
(bracket -14 and -48 ms) but it measures +74 and +57 ms. `predict_e18b`'s flip
formula put the sign change at 3.1 GiB/s, yet the arm is already A at 9.2 GiB/s,
so the real flip bandwidth is above 9 GiB/s, not 3. The calibrated alpha-beta
model under-predicts strategy B's boundary penalty for the seed-regenerated arm
at large d by roughly 90 to 105 ms; the mirrored arm it predicts correctly. That
is a concrete model-refinement result, not a null.

## E18b, the throttle sweep: not obtained here

E18b would extend the range below 9 GiB/s by `tc`-throttling the socket. It
could not run: RunPod containers have no NET_ADMIN, so `tc` is unavailable
(confirmed against the create API and an open RunPod feature request). The sweep
degraded to its socket-native point alone, which duplicates the 2x8 baseline, so
nothing new was measured. The frozen G4 predictions were written before that was
known and survive in the session log; the throttled points (10 Gbit, 1 Gbit) and
the InfiniBand point both need a provider with real VMs and working IB. Crusoe was
that provider; it has no 16xA100 capacity (checked 2026-08-25), so docs/13 is closed
and neither point has a path on A100 today. The socket-native result stands on its
own; nothing here is waiting on them.

## Session, honestly

Five clusters. The first four were lost to bugs found only at 16 devices or on a
real two-node fabric, each fixed before the next: the multi-slice mesh
(`jax.make_mesh` rejects it), a `timeout` wrapping a shell function, a dropped
set of shell helpers, a `$2`-before-assignment parse, an unpushed commit, the
invariance population (16 is half an antithetic pair on 16 devices), and the
invariance tolerance (the antithetic B arm's near-cancellation floors at ~5e-4
over socket, not the ~1e-7 NVLink NCCL holds; relaxed to 2e-3 for that arm
only). Cluster5 (this one) ran clean to `E18_SESSION_DONE`. Total RunPod cluster
spend for E18, all five, about $20; nothing left billing.
