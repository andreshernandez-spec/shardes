# E18 L1: the NCCL smoke on one 2-GPU pod. A100, 2026-08-22

docs/10 section 4, L1. One community pod with 2x A100-SXM4-80GB (NV12
between them), `l1.sh` at commit 62d69a3 with `e18-l1.yaml` (the rehearsal
shapes: d=64, N=16, batch 4, seq 8, warmup 1, repeats 3). Two runs of the
same preflight and driver: `1x2`, one process over both GPUs (the anchor,
writes the invariance reference), then `2x1`, two processes with one GPU
each through `jax.distributed`, which is the code path the cluster uses
across hosts. Every gate passed; nothing was retried.

What L1 retires: GPU distributed init over NCCL, the library's collectives
under two processes, `CUDA_VISIBLE_DEVICES` splitting, the preflight marker
and reference handling on real hardware, the driver's process-0-only
writing. What it cannot retire: the host boundary itself (both processes
are on one box, over NVLink), so nothing here says anything about the
inter-node fabric. That is the cluster's preflight.

## Session

Pod `tll2vwsyw53r40`, 216.249.100.66:22372, image
`runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404`, driver 595.71.05, jax
0.11.1 (jax-cuda12-plugin 0.11.1, nvidia-nccl-cu12 2.31.2), numpy 2.5.2,
scipy 1.18.1, python 3.12.3. Launched 21:23:50Z, `L1_DONE` at 21:29:41Z:
5m51s end to end. Harvested, then deleted at ~21:31Z.

One incident before it: a first pod (`wemenbdajcte9e`, 21:09-21:21Z) was
created with a public key typed from memory rather than read from
`~/.ssh/id_runpod.pub`, so every direct login was refused and the proxy
path (interactive shell only) could not repair it. Deleted and recreated;
about 12 minutes lost. Uptime for both pods ~22 min at $2.78/h, about
$1.0; the 21:00Z billing bucket was not yet closed at harvest time.

The cell records say `dirty_worktree: true`: the dirt is `l1.sh` (copied
to the pod by scp before it was committed as 57bb3c7, byte-identical) and
the two per-process logs. No source file differed from 62d69a3.

## Preflight

| | 1x2 (one process) | 2x1 (two processes) |
|---|---|---|
| identity | 2 x A100-SXM4-80GB, 1 process | 2 x A100-SXM4-80GB, 2 processes, 1 local device each |
| all-reduce 8 B | 312 us | 158 us |
| all-reduce 1 KiB | 311 us | 159 us |
| all-reduce 1 MiB | 316 us | 219 us |
| all-reduce 100 MiB | 696 us | 596 us |
| alpha, beta (fit) | 312 us, 254 GiB/s | 158 us, 223 GiB/s |
| invariance vs 1x2 | reference written | rel 0 on all four arms (bitwise) |
| warm cells (d=512, N=256) | 8.1 / 6.3 / 13.9 / 13.2 s | 8.1 / 6.5 / 12.9 / 13.1 s |

Two readings. The cross-process update is bitwise equal to the
single-process one, as NCCL was in the CPU-vs-GPU comparison before
(Gloo on the CPU rehearsal reassociated at ~2e-7). And the small-payload
time is LOWER with two processes than with one: the single-process ladder
sits at a flat 312 us up to 1 MiB, the two-process one at 158 us. So the
ladder's "alpha" in single-process mode is dominated by dispatching a
two-device executable from one process, not by the link; the fit's alpha
is therefore a per-topology dispatch floor, not a link latency, and must
be read as such when the cluster's 1x8 and 2x8 ladders are compared. The
bandwidth term is consistent across both (223-254 GiB/s for a 2-device
sum over NV12; not a cluster-relevant number).

## Driver cells

| arm | how | 1x2 median | 2x1 median |
|---|---|---|---|
| seed_regenerated | A | 1.63 ms | 1.48 ms |
| seed_regenerated | B | 1.39 ms | 1.19 ms |
| mirrored_lr1 | A | 1.16 ms | 0.68 ms |
| mirrored_lr1 | B | 1.18 ms | 0.83 ms |

Tiny shapes, sub-ms generations; these are a smoke, not a measurement.
The resume path was not exercised here (it was at L0).

**`seconds_all[0]` is compile-scale in every cell (5 to 14 s) despite
`warmup: 1`.** The first call after the warmup generation compiles again,
most likely because the state one generation returns has a different
abstract signature from `es.init`'s (weak-type promotion after a first
step is the usual cause; not verified here). Only the second timed call
onward is steady. The CPU rehearsal records show the same (1.5 to 1.8 s
first, then ms). `e18.yaml` has `warmup: 3`, so the
campaign's timed repeats are all steady; any analysis of L1 records must
drop `seconds_all[0]`, and the medians above already do by construction
(3 repeats, one outlier).
