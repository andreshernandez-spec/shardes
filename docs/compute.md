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

> **REFRESHED 2026-08-02, superseding the DWS-first advice below it.** Three things learned
> since this page was written change the answer, and only one of them is a price move.
>
> **1. The sweep driver is resumable and writes one file per configuration.** That is the big
> one. When this page was written nothing could survive an interruption, so it reached for
> on-demand. A preemption now costs **one configuration**, not the run, which makes spot and
> preemptible capacity rational rather than reckless.
>
> **2. Kaggle TPU v5e-8 works, and is queue-bound rather than slow.** 8 chips, free, verified
> end to end (`docs/06` T1). But the batch queue measured 2.5 h and then 4 h, against a
> 9-hour session cap and ~20 TPU-h/week. The compute is fine; the **calendar** is the problem,
> and no amount of money shortens a queue you are not paying for.
>
> **3. The interconnect is part of the result, not part of the invoice.** M1, M2 and M5
> measure communication volume and scaling across devices, and the A/B contraction crossover
> is the headline. On PCIe A100s rather than NVLink SXM, strategy B's model-size all-reduce
> behaves differently and the crossover contour becomes a property of the rental. **Require
> SXM/NVLink and put the topology in the results caption.** This is the reason not to take the
> cheapest marketplace price.
>
> Approximate, for **8x A100 80 GB over 4-6 hours**, which is the shape of the Phase 2 sweep:
>
> | option | ~$/hr for 8 | 4-6 h | lead time | interconnect |
> |---|---|---|---|---|
> | Vast.ai / spot marketplaces | $5-9 | $20-55 | minutes | **varies, often PCIe** |
> | **RunPod on-demand SXM** | **$9.5-12** | **$40-70** | **minutes** | NVLink |
> | Modal (serverless, per-second) | ~$20 | $40-80, idle free, $30/mo credit | none | NVLink |
> | CoreWeave 8x A100 | $21.6 | $86-130 | minutes | NVLink |
> | GCP `a2-ultragpu-8g` DWS | $19.2 | ~$115 | **days to weeks (quota)** | NVLink |
> | Colab Pro+ | n/a | $50/mo | minutes | **1 GPU, or a 1-chip TPU** |
>
> **Colab cannot do this at all** and is listed only so nobody re-proposes it: its CLI offers
> `TpuV5e1`/`TpuV6e1`, single chip, and one GPU per runtime. Phase 2 needs 8 devices in one
> session. Paying there buys a faster device, and the measurement needs *more* devices.
>
> **Do not start with GCP if speed is the point.** The quota process is still the single
> biggest scheduling risk on this page and can take longer than the TPU queue it would be
> replacing.
>
> ### The decision, as of 2026-08-02
>
> **RunPod on-demand 8x A100 SXM, ~$40-70 for the sweep.** No quota, minutes to start, SSH so
> there is no session cap and no one-shot-per-attempt problem, NVLink, and CUDA, which is the
> platform this stack is most proven on. `tests/gpu` passes on real GPUs and the TF32 trap is
> already handled, so the class of surprise the TPU produced does not recur.
>
> **Modal is the live alternative**, not an also-ran. It is script-first, which is what the
> driver already is; per-second billing with scale-to-zero means paying for compute rather
> than for the hours spent looking at it. If the sweep is 2-3 h of real compute, Modal likely
> wins on the bill actually paid rather than the rate quoted.
>
> **A100, not H100.** Every Phase 2 measurement except M4 is *relative* across device counts.
> H100 roughly doubles the cost to answer nothing extra. Reserve it for M4's absolute
> throughput comparison against EGGROLL's published figures, as a separate short booking.
>
> **Calibrate free before booking.** Kaggle's GPU tier queues in seconds. A timing run at real
> shapes on 1-2 devices tells you whether the sweep is 2 h or 12 h, which is the difference
> between a $50 booking and a $200 one.
>
> ### T1 stays in play, as a parallel track rather than the critical path
>
> Renting does not retire the TPU. It changes what the TPU is *for*: **the paid node is the
> critical path, and T1 runs beside it without blocking anything.** A queued kernel costs
> nothing to leave queued, so submit TPU work and carry on with the next phase rather than
> waiting on it.
>
> That is worth having for two reasons beyond cost. A TPU result is a **second platform**, and
> a scaling claim reproduced on two architectures is much harder to dismiss than one tuned to a
> single node. And the TPU already earned its place: it caught the bf16 matmul default that
> would have corrupted the sweep on **any** hardware with an aggressive precision default,
> including the A100 that would otherwise have been the first to run it.

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
