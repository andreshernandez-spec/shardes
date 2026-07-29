# Compute

How to get GPUs for each phase, what it costs, and how not to waste it.

> **Superseded for benchmarking.** This file covers *development-time* compute. The
> benchmark campaign now routes through free tiers — Kaggle's TPU v5e-8 (eight chips, $0)
> and TPU Research Cloud — with one paid GPU session for the cross-platform comparison.
> See **`docs/06-benchmark-runbook.md`**, which supersedes the "Recommendation" section
> below. The CPU-simulated-device guidance here is unchanged and still the most important
> cost lever in the project.

Prices are as researched on **28 July 2026** and move fast — re-check before booking.
Everything below is single-node multi-GPU. Multi-node is a different and much larger
problem (`jax.distributed.initialize()`, InfiniBand tuning, node failure handling) and is
not needed for any gate in `PLAN.md`.

---

## The single most important thing on this page

**Develop and test all sharding logic on CPU with simulated devices.**

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/
```

JAX will report 8 devices. `Mesh`, `NamedSharding`, `PartitionSpec`, `shard_map`,
collectives, `psum` placement, device-count invariance — all reproduce faithfully. What it
does *not* model is interconnect bandwidth or latency, so every **timing** claim needs real
hardware, but every **correctness** claim does not.

Roughly 90% of Phase 1 happens here, at zero cost. Renting a GPU to debug a `PartitionSpec`
is the most common way to waste money on a project like this.

---

## What each phase actually needs

| Phase | Hardware | Duration | Notes |
|---|---|---|---|
| 0 | 1 GPU. The local RTX 3080 covers it, see `docs/06-benchmark-runbook.md` T2 | ~1 day of runtime | Sweep is embarrassingly serial; run overnight |
| 1 | CPU-8 simulated, plus 1–2 GPUs to confirm | hours of GPU time total | The 2-GPU confirmation of `test_device_invariance` is the one that matters |
| 2 | **8 GPUs, one node** | 4–6 h | The real booking |
| 3 | 8 GPUs, one node | 4–6 h | Second session, or the tail of Phase 2's |

Phase 0 costs nothing now that it runs on the local 3080. Renting a GPU for it would be a
mistake.

---

## Answering the direct question: GCP or NVIDIA?

**Short version: GCP works and there is a specific product for exactly this shape of job,
but check your quota today. NVIDIA's own clouds are the wrong fit for a 6-hour individual
burst. A neocloud is the lowest-friction option if you have no GCP credits.**

### GCP

The relevant product is **Dynamic Workload Scheduler (DWS)**, which exists precisely for
short bursts of accelerator capacity without a reservation. Two modes:

- **Flex-start** — you submit, it queues, it runs when capacity appears. Cheapest.
- **Calendar mode** — you book a specific window in advance. Slightly more, but you know
  when you're running. Supports 8-GPU shapes only, which is exactly what you want.

DWS pricing (8-GPU nodes, per hour):

| Machine | GPUs | Flex-start | Calendar |
|---|---|---|---|
| `a2-highgpu-8g` | 8× A100 40GB | $16.00 | — |
| `a2-ultragpu-8g` | 8× A100 80GB | $19.20 | — |
| `g4-standard-384` | 8× RTX PRO 6000 | $18.00 | — |
| `a3-highgpu-8g` | 8× H100 80GB | $38.32 | $41.60 |
| `a3-megagpu-8g` | 8× H100 80GB | — | $44.00 |
| `a3-ultragpu-8g` | 8× H200 | — | $59.36 |
| `a4-highgpu-8g` | 8× B200 | — | $90.22 |

For contrast, plain on-demand `a3-highgpu-8g` is **$88.48/hr** — and note that it's the
*only* A3 configuration available on-demand; smaller A3 SKUs require Spot or Flex-start.
Spot H100 runs roughly $2.25/GPU/hr (~$18/hr for 8) with 30-second preemption notice.

**The one thing that will bite you: quota.** A GCP project with no history has zero H100
quota, and approval takes days to weeks. This is the single biggest scheduling risk in the
whole plan. **Request `NVIDIA_H100_GPUS` (or A100) quota during Phase 0**, weeks before you
need it. If it's declined, you've lost nothing and you fall back to a neocloud.

### NVIDIA

Three different things get called "NVIDIA cloud" and none of them is a great fit here:

- **DGX Cloud** — enterprise-tier, multi-node InfiniBand H100 clusters with the NVIDIA AI
  Enterprise stack bundled. Sales-gated, no public free tier, quoted from ~$15/hr per DGX
  and up. Built for Fortune 500 training runs, explicitly not for individuals doing a
  6-hour experiment.
- **DGX Cloud Lepton** — a GPU *brokerage*: you request compute, NVIDIA allocates it from a
  curated set of neoclouds (CoreWeave, Lambda, and others) and handles billing. Real
  advantages if you want one API across providers. The friction points that keep coming up
  are NVIDIA-controlled allocation, a curated rather than complete provider set, and less
  pricing transparency than going direct. For one 8-GPU node for six hours, the brokerage
  layer adds nothing you need.
- **Brev** (NVIDIA-owned) — ready-to-use GPU environments with "Launchables", preconfigured
  images on top of other clouds. Genuinely usable by an individual and the most plausible
  NVIDIA-branded option, but it's a convenience layer over the same underlying capacity.

There's a small narrative benefit to using NVIDIA's own stack if NVIDIA is a target
employer, but it shouldn't drive the technical decision, and "I ran a JAX scaling study on
8×H100" reads the same regardless of who billed you.

### Neoclouds

Lowest friction: no quota process, per-minute or per-second billing, often no egress fees.
Roughly, for H100 on-demand: RunPod ~$2.00–2.70/GPU/hr, Lambda ~$4.00/GPU/hr, specialist
clouds generally $2.50–$3.50/GPU/hr. So an 8×H100 node is roughly $16–32/hr.

---

## Recommendation

**Do the scaling study on 8× A100 80GB, not H100.**

`a2-ultragpu-8g` on DWS flex-start is **$19.20/hr → ~$115 for a 6-hour session**, about
half the H100 price. Strong and weak scaling curves, the contraction crossover, memory
scaling, and communication volume — M1, M2, M3, M5, M6 in `docs/03-phase2-benchmarks.md` —
are all about *relative* behaviour across device counts. A100 answers every one of them.

Reserve H100 for one thing only: if you want an absolute throughput number comparable to
EGGROLL's published H100 figures (M4). That's a short, targeted run, not the whole sweep,
and it can be a second booking once you know the sweep works.

Decision rule:

1. **Have GCP credits, or want it on GCP?** → request A100 quota now, book
   `a2-ultragpu-8g` via DWS flex-start. ~$115 for the session.
2. **No credits, want it working this week?** → RunPod or Lambda, 8×H100 or 8×A100,
   ~$130–190 for six hours, no quota process.
3. **Want NVIDIA specifically?** → Brev, and accept that you're paying a convenience
   premium over (2) for approximately the same machine.

Budget the whole project at **$300–500** of compute across all phases, assuming the dress
rehearsal in `docs/03-phase2-benchmarks.md` is taken seriously. Without the rehearsal,
assume double.

---

## Setup runbook

Single-node multi-GPU JAX. Nothing exotic is required.

```bash
# Install. JAX >= 0.11; do not pin below.
pip install -U "jax[cuda12]"

# Sanity check — must print 8 devices before anything else runs.
python -c "import jax; print(jax.__version__); print(jax.devices()); print(jax.device_count())"
```

If `jax.devices()` shows fewer GPUs than the box has, stop and fix that first — it is
almost always a driver/CUDA mismatch, and no amount of sharding code fixes it.

```python
from jax import shard_map                       # NOT jax.experimental.shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, axis_names=("pop",))
```

**No `jax.distributed.initialize()`** on a single node. It's for multi-host only, and
calling it unnecessarily produces confusing hangs.

Useful environment flags:

```bash
# Simulated devices on CPU — the development default.
export XLA_FLAGS=--xla_force_host_platform_device_count=8

# Don't preallocate 75% of every GPU; useful when profiling memory (M6).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Deterministic collectives, for the device-invariance check on real hardware.
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_deterministic_ops=true"
```

Container: NVIDIA's NGC JAX image is the least surprising base if the host has a matching
driver. Build the image **before** the rented session, push it, and boot one cheap
single-GPU instance from it to confirm it starts. Debugging a `pip install` on an 8-GPU
node bills at 8-GPU rates.

---

## Cost discipline

- Set a billing alert before the first instance boots. On every provider.
- **Never leave an idle multi-GPU node running.** Put a hard shutdown in the driver script:
  it should power the box off when the sweep completes or the wall-clock cap trips, not
  wait for you to notice.
- Write results to durable storage (GCS bucket, or `rsync` out) as each configuration
  completes, not at the end. A preempted spot instance at hour five with everything in
  `/tmp` is a total loss.
- Make the driver resumable: re-running skips completed configurations. This turns
  preemption from a disaster into a delay, and makes spot pricing genuinely usable.
- Log the environment (GPU model, driver, CUDA, JAX version, commit SHA) into the results
  directory automatically. Reconstructing it afterwards is never accurate.
