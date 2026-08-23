# E13 clean rerun: the four ES arms and the frozen-embedding ablation at a clean SHA. A100, 2026-08-22

The 08-17 campaign (`../e13-a100-2026-08-17/`) stamped its worktree dirty. This is the
same experiment re-run from a committed tree: the four ES arms (`pilot.yaml`,
`pilot-lr1.yaml`, `pilot-lr4.yaml`, `pilot-lr16.yaml`) and the frozen-embedding
ablation (`pilot-lr1-frozen-embed.yaml`), seeds 0-2, 500 generations each, 15 runs.
Driver `e13_campaign.sh` (refuses a dirty tree), collected with `e13_harvest.sh`. Code
at 8d06d64, jax 0.11.1, transformers 5.15.1, `runpod/pytorch:1.0.2-cu1281` image, three
community A100-SXM4-80GB pods in parallel: `seed` on one host, `lr1 lr1-frozen-embed`
and `lr4 lr16` on two pods of a second host (EPYC 7742, same image and wheels). GRPO was
not re-run; `plot_e13.py` reads its evals from the 08-17 directory.

Final held-out eval over seeds 0-2, mean [min-max] (`tb3.py` emits the same numbers):

| arm | eval reward | 08-17 |
|---|---|---|
| mirrored-seed (full rank) | 0.156 [0.151-0.162] | 0.157 [0.154-0.162] |
| lr1 | 0.157 [0.154-0.160] | 0.155 [0.152-0.158] |
| lr4 | 0.154 [0.149-0.157] | 0.155 [0.149-0.160] |
| lr16 | 0.155 [0.152-0.158] | 0.153 [0.151-0.156] |
| lr1, embedding frozen | 0.155 [0.153-0.158] | 0.154 [0.150-0.157] |

Quality reproduces inside seed noise. Timing does not, for one arm:

| arm | s/update, steady (updates 2-499), median over seeds | 08-17 |
|---|---|---|
| full rank | 4.45 (4.41, 4.45, 4.45) | 4.29 |
| rank 1 | 2.98 (2.98, 2.99, 2.97) | 2.40 |
| rank 4 | 2.91 (2.94, 2.89, 2.91) | 2.78 |
| rank 16 | 2.83 (2.85, 2.83, 2.83) | 2.69 |
| rank 1, frozen | 2.97 (2.97, 2.97, 2.94) | 2.44 |

Full rank and ranks 4 and 16 moved 4-5%, which is host drift (the 08-17 campaign ran
every arm on one pod; this one used two hosts) plus jax 0.11.0 to 0.11.1. Rank 1 moved
24%, and now costs what rank 4 costs. Between the two campaigns sits a671dc6, which
pads the r=1 factors to a rank-2 dot so the TPU keeps them fused (rank 1 OOMed cells
rank 4 ran, `../../../phase2/results-cost-tpu-v5e8/README.md`). The pad is not
platform-conditional, so the A100 runs it too.

`probes/` settles the attribution on one host, one SHA (6d56af5), one jax, 30
generations each (`probes/results-a100-2026-08-23/`, diagnosis only):

| probe | s/update |
|---|---|
| full rank | 4.32 |
| rank 1, as committed (padded) | 2.89 |
| rank 1, pad switched off by a local edit, reverted after | 2.49 |

So on the A100 the pad costs rank 1 16% per update. Against full rank, rank 1 as
committed uses 33% less time per update (1.50x updates per second); unpadded it would
use 42% less (1.73x), which is the 08-17 campaign's 44% / 1.78x within host drift. The
paper cites the committed program. Whether the pad should apply only on TPU is a
library decision, not made here; if it changes, every post-08-19 A100 rank-1 number
(this campaign, E15, E16) moves with it.

Compilation-scale generations 0 and 1 take 322-590 s and 343-635 s across the 15 runs
and are excluded from every steady-state figure, as before. One incident: the first
launch of all three pods died in generation 0 on `apply_chat_template requires jinja2`
(a transformers 5.15.1 dependency the bootstrap did not install); jinja2 was added and
the campaigns relaunched from scratch, so no run here was resumed.

Costs: three pods, 20:19 to 00:05 / 00:50 / 01:45 UTC including the failed launch and
the probes, about 13.7 pod-hours, $19.
