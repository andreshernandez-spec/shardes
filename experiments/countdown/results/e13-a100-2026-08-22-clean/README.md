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
| lr1 | 0.156 [0.155-0.157] | 0.155 [0.152-0.158] |
| lr4 | 0.154 [0.149-0.157] | 0.155 [0.149-0.160] |
| lr16 | 0.155 [0.152-0.158] | 0.153 [0.151-0.156] |
| lr1, embedding frozen | 0.150 [0.147-0.154] | 0.154 [0.150-0.157] |

Quality reproduces inside seed noise.

| arm | s/update, steady (updates 2-499), median over seeds | 08-17 |
|---|---|---|
| full rank | 4.45 (4.41, 4.45, 4.45) | 4.29 |
| rank 1 | 2.61 (2.61, 2.64, 2.57) | 2.40 |
| rank 4 | 2.91 (2.94, 2.89, 2.91) | 2.78 |
| rank 16 | 2.83 (2.85, 2.83, 2.83) | 2.69 |
| rank 1, frozen | 2.60 (2.60, 2.59, 2.61) | 2.44 |

Full rank and ranks 4 and 16 are 4-5% off the 08-17 campaign, which is host drift (that
campaign ran every arm on one pod; this one used several) plus jax 0.11.0 to 0.11.1.

**The rank-1 arms were measured twice.** Run first at 8d06d64 they came out 24% slower
than 08-17 (2.98 and 2.97 s), which is a671dc6: it pads the r=1 factors to a rank-2 dot
so the TPU keeps them fused (rank 1 OOMed cells rank 4 ran,
`../../../phase2/results-cost-tpu-v5e8/README.md`), and it padded on every platform.
A same-host probe priced the pad on the A100 at 16% per update (`probes/`, below), so
030732d made the pad TPU-only and the two rank-1 arms were re-measured under it. The
timings above are that rerun; the padded measurements are not kept, since they are not
the released program on this platform. The other three arms are unaffected by the pad
(nothing else in the diff touches the update path) and keep their 8d06d64 records, so
this directory carries two commits by design.

Rank 1 is now the cheapest arm, 41% below full rank, where under the pad it was slower
than ranks 4 and 16. The cost surfaces say the cheapest perturbation should also be the
cheapest update, so that ordering is a consistency check the padded numbers failed.

The rerun's rank-1 curves are not the padded runs' curves: separately compiled programs
flip near-tied greedy tokens and the sampled trajectories diverge from there (the
manuscript's real-hardware invariance paragraph). Held-out reward is unmoved,
0.156 [0.155, 0.157] against the padded 0.157 [0.154, 0.160].

`probes/` settles the pad's cost on one host, one SHA, one jax, 30 generations each
(`probes/results-a100-2026-08-23/`, diagnosis only):

| probe | s/update |
|---|---|
| full rank | 4.32 |
| rank 1, pad off (what the A100 now runs) | 2.49 |
| rank 1, pad on (what the TPU runs) | 2.89 |

Scope of the pad change, audited over every committed result record: of 1081 A100
timing records only four stamp a commit at or after a671dc6, and none of them is a
rank-1 cell. The A100 cost surface's 40 `mirrored_lr1` cells stamp c93cf5d, four days
before the pad, so they measured the unpadded program all along and describe the
released code again now; the v5e's 40 stamp 08-19 commits and are padded, which the
TPU still is. Alignment results (E15, E16) are cosines, and the pad is numerically
invisible, so they are unaffected either way.

Compilation-scale generations 0 and 1 take 322-590 s and 343-635 s across the 15 runs
and are excluded from every steady-state figure, as before. One incident: the first
launch of all three pods died in generation 0 on `apply_chat_template requires jinja2`
(a transformers 5.15.1 dependency the bootstrap did not install); jinja2 was added and
the campaigns relaunched from scratch, so no run here was resumed.

Costs: three pods, 20:19 to 00:05 / 00:50 / 01:45 UTC including the failed launch and
the probes, about 13.7 pod-hours, $19.
