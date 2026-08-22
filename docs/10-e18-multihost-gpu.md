# 10 — E18: the contraction crossover across a real host boundary (2x 8xA100)

Planned 2026-08-22, before any code or rental. The governing constraint, stated
by Andres: be as careful as possible; do not run for 30 minutes only to
discover a bug. The design answer is a risk-retirement ladder: every layer of
the stack is proven on something free or nearly free before the cluster
exists, and the cluster session itself opens with a scripted preflight that
can abort for a few dollars instead of failing at hour two.

## 1. The question, and why GPUs

The E5-deepened design (docs/05) split the multi-host question in two. The
depth-in-D half (8 to 64 chips on one torus) waits for TRC. This is the other
half: the host-boundary CLIFF. Inside an A100 node, collectives ride NVLink
(600 GB/s class); between nodes they ride InfiniBand or Ethernet (roughly
12-50 GB/s, sometimes worse). One to two orders of magnitude, exactly at the
boundary. Strategy B pays that cliff on a model-sized all-reduce every
update; strategy A pays it on scalars. If the practical rule ("partial
contraction for expensive perturbations, scalar gathering for cheap ones
once collectives cross hosts") is real anywhere, this is where it appears
first.

## 2. Frozen hypotheses (directional, committed here)

With P the parameter count, BW_intra the NVLink all-reduce bandwidth and
BW_inter the measured inter-node bandwidth:

- **H1 (the controlled boundary effect).** At the SAME D=8, moving from one
  node (8 GPUs) to a 2x4 split (4 GPUs per node) leaves strategy A's
  generation time approximately unchanged and increases strategy B's by
  approximately 4P * (1/BW_inter - 1/BW_intra). This is the cleanest test in
  the design: same device count, same shapes, boundary on vs off; nothing
  else moves.
- **H2 (seed-regenerated survives the cliff).** At D=16, B still beats A for
  the seed-regenerated arm: A's replicated contraction cost is constant in D
  and large, B's shrinks as 1/D, and the all-reduce penalty is bounded by H1's
  delta. A sign flip here falsifies the transfer of the single-node result.
- **H3 (rank 1 flips to A).** At D=16, A beats B for the rank-1 arm: B's
  compute saving is small at these shapes (M3 showed near-parity on NVLink)
  and the boundary penalty lands on top. This is the half of the practical
  rule the single-node data cannot show.
- **H4 (the calibrated model predicts it).** An alpha-beta model per link
  class, calibrated ONLY from the preflight microbenchmarks plus the
  committed single-node sweep, predicts the sign and rough magnitude of
  t_B - t_A for every D=16 cell before those cells run.

A gentle or absent effect under a strong fabric is a finding, not a failure:
the fabric is measured first and every claim is stated relative to the
measured BW_inter (the fabric a rented cluster provides varies from IB to
plain Ethernet, and the decision rule must be portable across that range).

## 3. What runs (cells, all frozen before the session)

Workload: the phase-2 synthetic block, unchanged, so D=16 extends the
committed D<=8 curves rather than starting a new family. Same protocol:
matmul precision highest, warmup 3; repeats 7 (not 5: network jitter), IQR
recorded; interleaved A/B per cell.

- Arms: `seed_regenerated` and `mirrored_lr1` (the bracketing pair), A and B.
- (N, d) cells, chosen from the committed M3 GPU diagram at execution time
  and named in the config before launch: one B-favored (seed_regenerated,
  d=2048), one near-parity (mirrored_lr1, d=512), one A-leaning
  (mirrored_lr1, d=2048).
- Topologies: D=8 one-node (reproduces the existing sweep: the sanity
  anchor), D=8 split 2x4 (H1), D=16 (H2/H3). Strong scaling at the frozen
  cells; weak-scaling variants only if the session has slack (recorded
  either way).
- Component isolation, barrier.py-style, per topology: the fitness gather,
  the model-sized psum at {1 KB, 1 MB, 4P bytes}, and the contraction alone.
  These are simultaneously the calibration inputs for H4 and the published
  decomposition.

Cell count: 2 arms x 2 hows x 3 cells x 3 topologies = 36 timed cells, plus
~12 component cells. At the synthetic block's compile-and-run times (~1-3 min
per cell at these shapes), the campaign is comfortably under 3 hours.

## 4. The risk-retirement ladder

**L0 — CPU multi-process rehearsal (free, local).** The FULL driver, two
processes x 4 simulated host devices each (`jax.distributed.initialize`,
local coordinator, `XLA_FLAGS=--xla_force_host_platform_device_count=4`,
`JAX_PLATFORMS=cpu`), tiny block. Gates, all scripted:
  - the update from 2x4 multi-process equals the single-process D=8 update
    under fixed fitnesses to tolerance (the invariance test, across the
    process boundary — the most important test in the repo, extended);
  - only process 0 writes; per-cell resume works across a killed process;
  - the driver runs degenerate single-process too (same code path L0-L3).
L0 must pass before anything is rented. It retires: distributed init, mesh
construction over processes, config plumbing, resume, writer discipline, and
the library's multi-host correctness.

**L1 — single-host NCCL smoke (~$3, one cheap pod).** Same driver, two
processes x 1 GPU each on a 2-GPU pod. CPU cannot exercise the NCCL backend;
this does, without multi-node procurement. Retires: GPU distributed init,
NCCL collectives through our collectives, CUDA-visible-devices splitting.
Budget one hour; a hang here costs $3, not $30.

**L2 — cluster preflight (first ~15 minutes of the rented session,
scripted, abortable).** `preflight.py`, exits non-zero on any failure, and
the campaign driver refuses to start without its pass marker:
  1. Both nodes: identical GPU name/count, driver, jax/jaxlib versions;
     hostnames differ (a provider handing two VMs on one box would show up
     here and in step 3's suspiciously high bandwidth).
  2. Distributed init across nodes with a hard timeout (a hang is the most
     likely first-contact failure: coordinator port not open between nodes,
     or NCCL picking the wrong interface — NCCL_SOCKET_IFNAME recorded).
  3. Link microbenchmark: psum at {8 B, 1 KB, 1 MB, 100 MB} intra-node and
     inter-node; fits alpha (latency) and beta (bandwidth) per link class;
     writes `calibration.json`. Go/no-go: inter-node must complete and the
     measured fabric is recorded whatever it is; abort only on errors or on
     evidence the topology is not what was rented.
  4. The invariance gate on the real fabric: D=8 one-node vs D=16 update
     under fixed fitnesses, within tolerance.
  5. One warm cell per (arm, how) at the smallest shape, with a per-cell
     time cap, confirming compile times are in budget.
If preflight fails: terminate the cluster. Cost of discovery: ~$8.

**L3 — predictions, then the campaign.** `predict.py` (committed before the
session) reads `calibration.json` plus the committed single-node results and
writes `predictions.json` — sign and estimate of t_B - t_A for every D=16
cell — BEFORE any D=16 ES cell runs; the driver enforces the order. Then the
36 cells, resumable, one JSON per cell, rsynced back to the workstation
every few minutes so a teardown never loses more than the cell in flight.

## 5. Procurement

Two paths, decided at execution time by what has capacity:

- **RunPod Instant Clusters** (REST v2 has /v2/clusters; the curated MCP
  tools do not expose it, so creation is via the console or a direct API
  call). This is the preferred path: it advertises high-bandwidth east-west
  networking, and 2x 8xA100-80GB should land near $25-30/h.
- **Fallback: two SECURE pods, same data center, Global Networking.** The
  overlay fabric is slow (order 1-10 Gb/s) — which still demonstrates the
  cliff, in exaggerated form. Acceptable only with the fabric measured and
  reported as what it is; the calibrated model absorbs it, the prose must
  not present it as a well-built cluster.

Total budget: L1 ~$3; cluster session preflight + calibration + campaign +
slack = 4 h x ~$28 ≈ $115, abort path ~$8. Hard cap: terminate at 5 h
regardless; resume in a second session costs only the queue-free re-setup.

## 6. Failure modes and their mitigations

| failure | caught by | cost if it slips through |
|---|---|---|
| distributed init hangs (ports, NCCL iface) | L1; L2 step 2 timeout | ~$8, abort |
| library wrong across processes | L0 invariance gate | $0 |
| "two nodes" are one machine | L2 steps 1+3 | ~$8, re-procure |
| driver bug (resume, writer, config) | L0 runs the real driver | $0 |
| compile blowup at D=16 | L2 step 5 time cap | ~$10 |
| results lost at teardown | continuous rsync | one cell |
| network jitter drowns the effect | 7 repeats + IQR; component isolation measures the delta directly | wider bars, still a result |
| fabric weaker than advertised | L2 step 3 records it; claims are relative to measured BW | reframed, not wasted |

## 7. Artifacts to build (all committed before any rental)

- `experiments/phase2/multihost/driver.py` — the sweep driver with
  `jax.distributed` init from env, process-0 writing, per-cell resume,
  degenerate single-process mode.
- `experiments/phase2/multihost/preflight.py` — section 4 L2, plus the
  psum microbenchmark it shares with the calibration.
- `experiments/phase2/multihost/predict.py` — H4, from calibration.json +
  committed single-node results only.
- `experiments/phase2/multihost/rehearse-cpu.sh` — L0, runnable by anyone.
- `experiments/phase2/multihost/e18.yaml` — cells, hypotheses header,
  protocol constants.
- `launch.sh` — per-node start commands (coordinator on node 0).

## 8. Order of work

1. Build everything in section 7; L0 green locally. (Free; no clock.)
2. PR with L0 evidence in the description. Andres reviews the hypotheses.
3. L1 on a $3 pod; result appended to the PR.
4. Procure (section 5), run L2+L3 in one sitting, terminate.
5. Harvest: analysis vs predictions.json, results README with the fabric
   characterization front and center, paper follow-up decision after.

Steps 1-3 need no further approval; step 4 waits for an explicit go with
the provider choice.
