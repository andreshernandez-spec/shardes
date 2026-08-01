# 06 — Benchmark runbook

The operational half of `docs/05-paper.md`. Which machine runs which experiment, how to get
it, and how not to lose a session.

---

## Compute tiers

| Tier | What | Devices | Cost | Limits |
|---|---|---|---|---|
| **T0** | Local / any CPU | 8 simulated | $0 | No timing validity |
| **T1** | **Kaggle TPU v5e-8** | **8 chips, 16 GB each** | **$0** | ~20 TPU-h/week, 9-h sessions |
| **T2** | **Local RTX 3080 Laptop, 16 GB, Ampere** | 1 | **$0** | Laptop thermals; single device, so no sharding |
| **T2′** | Kaggle GPU: P100 16 GB, or 2× T4 | 1–2 | $0 | ~30 GPU-h/week; Turing. Fallback for T2 |
| **T3** | TPU Research Cloud | 8–256 chips, per grant | $0 TPU + ~$10–20 VM/GCS | Temporary grant, ~30 days |
| **T4** | GCP DWS, `a2-ultragpu-8g` | 8× A100 80 GB, NVLink | $19.20/h | Needs quota |
| **T4′** | GCP DWS, `a3-highgpu-8g` | 8× H100 80 GB | $38.32/h flex-start | Only if matching EGGROLL's H100 numbers |
| **T5** | Vast/RunPod spot | 1–8 | ~$0.7–2/GPU-h | Reruns, host quality varies |

The routing principle: **every experiment goes to the cheapest tier that can answer it.**
Almost everything lands on T0–T3, which are free.

---

## T0 — CPU. Where most of the work happens.

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/
```

Eight simulated devices. `Mesh`, `PartitionSpec`, `shard_map` signatures, collective
placement, device-count invariance, communication *byte counts* — all reproduce faithfully.
Interconnect latency does not, so no timing claim comes from here.

If any Pallas kernels get written, `interpret=True` runs them on CPU — qwix's own CI does
exactly this for its TPU kernels. Between the two flags, every correctness question in this
project is answerable for free. **Rent only to measure.** More money gets burned debugging
at $2/hr than on the benchmarks themselves.

---

## T1 — Kaggle TPU v5e-8. The primary scaling platform.

Eight chips, free, ~20 TPU-hours/week, 9-hour session cap, no credit card. This carries
E2, E3, E4 (first pass), E8-TPU, E10, E11 — the bulk of the paper.

**Constraints to design around:**

- **16 GB HBM per chip.** Population sizes are memory-bound well below EGGROLL's H100
  figures. Choose model shapes so that `D ∈ {1,2,4,8}` all fit at the same per-device `N`,
  or the scaling curve compares different computations.
- **9-hour hard stop, ephemeral filesystem.** Anything not written out is gone.
- **Notebook execution, no SSH.** The library must be `pip install`-able from GitHub so the
  notebook is a thin driver, not a code dump. Commit SHA gets pinned and logged.
- **Weekly quota resets.** Plan `E4`'s grid as four ~5-hour chunks across two weeks rather
  than one heroic run.

**Session template:**

```python
# Cell 1 — pin and install
!pip install -q "git+https://github.com/<you>/shardes@<SHA>"
import jax; print(jax.__version__, jax.devices(), jax.device_count())   # expect 8

# Cell 2 — resume state from a Kaggle Dataset or GCS
# Cell 3 — run only configs not already present in results/
# Cell 4 — write results incrementally; push after EVERY config, not at the end
```

Make the driver **idempotent and resumable**: it reads which `(experiment, config)` pairs
already have results, runs only the rest, and writes each one out as it completes. A
9-hour session that dies at hour 8 should cost you one configuration, not everything.

---

## T2 — the local RTX 3080. Phase 0 only.

16 GB, Ampere. Runs **E1**, the estimator sweep. E1 is embarrassingly serial, needs one
device, and measures statistics rather than throughput, which is exactly the shape a
machine you already own handles best: no session cap, no weekly quota, no notebook driver,
and results land straight on durable disk.

It beats the Kaggle GPU tier on every axis E1 cares about. Same 16 GB as the P100, real
bf16 tensor cores that the Turing T4s don't have, and no 9-hour stop.

Two things to respect. It's a laptop part, so a 20-hour sweep is a sustained thermal load
and clock throttling will make wall-clock numbers meaningless. That's acceptable here
only because **no timing claim comes from E1**; wall-clock per estimate is context, not a
result. And it's a single device, so nothing about sharding is testable on it. Sharding
stays on T0.

### T2′ for Gate G1 criterion 2 — the two-GPU invariance check

**Status 2026-08-01: PASSED. 16 passed in 74.97 s on 2 x Tesla T4**, Kaggle, jax 0.11.0,
commit `e720c92`, `matmul precision: highest`. No skips. Run headlessly through the API with
`python experiments/phase1/kaggle/run.py t2prime`; the log is under `output/t2prime/`.

The reference it was checked against: `{'jax': '0.11.0', 'platform': 'cpu', 'device_kind':
'cpu', 'device_count': 8}`. That is the point of the exercise, a simulated 8-device CPU result
reproduced on hardware that shares none of its assumptions.

The check splits into two claims that want different hardware and different tolerances, and
`tests/gpu/test_device_invariance_gpu.py` keeps them apart:

| claim | needs | tolerance | status |
|---|---|---|---|
| one real GPU reproduces the CPU-8 simulated reference | 1 GPU | `1e-4`, loose: different kernels, different reduction trees | **passing** on the RTX 3080, 3 strategies × A and B |
| 2 GPUs give the same update as 1 | **2 GPUs** | `1e-5`, tight: only summation order changes | **passing** on 2x T4, 2026-08-01 |
| A and B agree over a real interconnect | **2 GPUs** | `1e-5` | **passing** on 2x T4, 2026-08-01 |

The second and third are the ones simulated devices *cannot* answer:
`--xla_force_host_platform_device_count` gives eight devices that share one memory space and
never actually communicate, so a collective that is wrong over a real interconnect still
passes there. That is the whole reason docs/02 asks for this before Phase 2.

**Set `jax_default_matmul_precision="highest"`.** The test fixture does it, but know why: an
Ampere GPU defaults to TF32 for matmuls, which is ~1e-3 relative and reads exactly like a
device-invariance failure. Also pass `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.

**Getting the code onto Kaggle.** The repo is public as of 2026-08-01:
`github.com/andreshernandez-spec/shardes`, Apache 2.0. Clone it, anonymously, and check out
the SHA the result should be attributed to.

**Clone, do not `pip install`.** Rehearsed on 2026-08-01, first against a `git archive` zip
and then against the public clone. The obvious install path does not work:

- `pip install "git+…@SHA"` (and `pip install shardes.zip`) installs the *package* and nothing
  else. `tests/` and `experiments/phase1/reference.json` are not inside `src/shardes/`, so they
  do not land, and `pytest tests/gpu` then reports **`no tests ran in 0.00s`** — a message that
  scrolls past looking like nothing went wrong.
- It also *builds* a wheel, which fetches `hatchling` from PyPI, for nothing.

A clone carries the tests, the reference and the SHA together, which is the whole point.
`PYTHONPATH=src` then needs no install step at all.

**Pin the SHA.** `main` moves. A T2′ result recorded against "HEAD" is not attributable, and
this check exists to be quoted in a gate.

**Internet must be on, for jax, not for us.** `src/shardes/contraction.py` does
`from jax import shard_map`, which is jax ≥ 0.8, and `pyproject.toml` floors it at 0.11.
Kaggle's image ships an older jax, so the upgrade is unavoidable and it is a ~500 MB CUDA
download. Enable internet in the notebook sidebar (needs a phone-verified account) *before*
starting the session. Running from source avoids the `hatchling` fetch, not this one.

Do the version check first and let it fail loudly. An outdated jax surfaces as a collection
`ImportError` twenty lines deep, which reads like a bug in `shardes`.

```python
# Cell 1 — Kaggle notebook, accelerator = GPU T4 x2
!pip install -q -U "jax[cuda12]>=0.11"
```

```python
# Cell 2 — preflight. Wrong answers here are cheap; wrong answers after a 40-minute run are not.
import os
SHA = "d1722ec"          # SET THIS to the commit under test. Do not copy it forward.
!git clone -q https://github.com/andreshernandez-spec/shardes.git /kaggle/working/shardes
os.chdir("/kaggle/working/shardes")
!git checkout -q $SHA && git log --oneline -1
os.environ["PYTHONPATH"] = "src"
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
!python -c "import jax; v=jax.__version__; assert tuple(map(int,v.split('.')[:2]))>=(0,11), f'jax {v} < 0.11'; \
from jax import shard_map; d=jax.devices(); print(v, len(d), d[0].device_kind); assert len(d)==2, f'{len(d)} devices, want 2'"
```

```python
# Cell 3 — the check itself.
!python -m pytest tests/gpu -m gpu -q -s        # expects 16 passed on 2 GPUs
```

Sanity-check the count. **16 collected** is the sign the checkout is intact; anything less
means the delivery is broken, not that the code is fine.

`experiments/phase1/reference.json` is committed and travels with the checkout, so the CPU-8
reference does not have to be regenerated on Kaggle — and must not be: the point is to carry
the *simulated* result to hardware that does not share its assumptions. The fixture **skips**
when it cannot find that file rather than failing, so a broken delivery shows up as
`6 skipped`, not red. Read the skip reasons; do not read the exit code.

Record the output of `test_report_the_environment` with the result. A green tick that names no
hardware is not evidence.

**Keep the notebook private.** T2′ runs on the personal Kaggle account
(`al252130@gmail.com`), which is not the `andreshernandez-spec` identity the repo is published
under; a phone number can only verify one Kaggle account, so this is not a thing to tidy up
later. It costs nothing: the result travels into the repo as the `test_report_the_environment`
output in a commit, so provenance comes from the commit, not from the notebook. Publishing the
notebook would be the one step that ties an unrelated identity to the project for no gain.

---

**T2′, the fallback:** Kaggle GPU, P100 (16 GB) or 2× T4, ~30 GPU-hours/week, if the
laptop is needed for something else or a config needs more memory headroom. T4s are
Turing: no useful bf16 tensor cores, no Hopper features. **They never appear in a
throughput claim.** Correctness and statistics only.

---

## T3 — TPU Research Cloud. The scaling-past-8 tier.

Free Cloud TPU quota granted to your GCP project, ready to use within minutes of accepting
an invitation. TPUs are free; you pay only for a small `n1-standard-2` driver VM and a GCS
bucket, which is minimal. The stated obligation is to share the work publicly through
peer-reviewed publications, open source code, or blog posts — precisely the plan.

**The timing trap, and it's the most important paragraph on this page.**

Invitations go out on a rolling basis, and the grant is **temporary** — historically a
30-day window, with a specific quota in a specific zone (e.g. "32 on-demand v4 chips in
us-central2-b"), and you cannot create TPU types you don't have quota for. The clock starts
when you accept.

So: **apply around week 10, not week 1.** This is the opposite of the GPU-quota advice in
`docs/compute.md`. Applying early and accepting before the code runs is the standard way to
burn the window. Have E5 and E12 ready to launch the day the quota lands.

In the application, describe the actual project — a sharded ES library, an open-source
artifact, a paper. That's exactly the profile the program funds.

**Also worth knowing:**

- TRC quota is not compatible with Vertex-based workflows; use the plain GCE/TPU-VM path.
- Preemptible capacity sees frequent interruptions. The same resumability discipline as
  Kaggle applies, doubly.
- Ask for what you need. A `v5e-32` or `v5e-64` grant is what makes E5's 64-device curve
  possible; a `v5e-8` grant just duplicates Kaggle.

```bash
gcloud compute tpus tpu-vm create shardes-bench \
  --zone=<ZONE_FROM_GRANT_EMAIL> \
  --accelerator-type=v5litepod-32 \
  --version=<TPU_SOFTWARE_VERSION>

gcloud compute tpus tpu-vm ssh shardes-bench --zone=<ZONE> \
  --worker=all --command="pip install -q 'git+https://github.com/<you>/shardes@<SHA>'"
```

Multi-host slices (>8 chips) do need `jax.distributed.initialize()` and run one process per
host. This is the one place the "single-node only" simplification breaks, so budget a day
for it and rehearse on a `v5e-8` first.

---

## T4 — GCP paid GPU. One session, cross-platform comparison.

`a2-ultragpu-8g` — 8× A100 80 GB, NVLink — at **$19.20/hr via DWS flex-start**. Six hours
is **~$115**. Runs E6, E7, E8-GPU, E9.

A100 rather than H100 because the GPU session's job is now the *NVLink-vs-ICI comparison*,
not an absolute throughput record. If a headline number matched to EGGROLL's published
H100 figures is wanted, add a short second session on `a3-highgpu-8g` ($38.32/hr
flex-start) with only E9 in it.

DWS has two modes: **flex-start** queues until capacity appears (cheapest), **calendar**
books a known window (slightly more, 8-GPU shapes only). For a scheduled benchmark day,
calendar is worth the difference.

**Request GPU quota during Phase 0.** A fresh project has zero A100/H100 quota and approval
runs days to weeks. Unlike TRC, there's no cost to having it early.

---

## Not losing the paid session

Rented multi-GPU time is the only irreversible cost here. The session is an *execution* of
a debugged plan, never a debugging session.

**Two weeks before** — GPU quota requested. TRC applied for.

**The week before** — full dress rehearsal at 1–2 GPUs with `N` reduced 100×. The local
T2 box covers the 1-GPU case; use T2′ or T5 if the rehearsal needs two devices to exercise
a collective:
- every configuration runs to completion,
- results written incrementally, one file per config,
- driver resumable — re-running skips what's done,
- hard wall-clock cap per config; exceeding it logs and moves on,
- `plot.py` runs end-to-end on rehearsal data and emits the final figures.

If the rehearsal doesn't produce publication-shaped figures from fake numbers, the real
session won't either.

**The day before** — build and push the container image, boot one cheap single-GPU
instance from it, confirm `jax.devices()` reports the GPU and the driver starts. Image
build failures on an 8-GPU box bill at 8-GPU rates.

**During** — priority order, so running out of time loses the least important measurement:
E6 (F1) → E7 (F2) → E8-GPU (F4) → E9 (TB1). Ordering matters more than duration.

**Always** — hard shutdown in the driver when the sweep finishes or the cap trips. Never
wait for a human to notice an idle 8-GPU node.

---

## Budget

| Item | Cost |
|---|---|
| T0 CPU | $0 |
| T1 Kaggle TPU v5e-8 (~60 h) | $0 |
| T2 local RTX 3080 (~20 h) | $0, electricity |
| T3 TRC — VM + GCS only | ~$10–20 |
| T4 one 6-h A100 session | ~$115 |
| T4′ optional H100 session for E9 | ~$120 |
| T5 reruns, spot | ~$20 |
| **Total** | **~$150 realistic, ~$300 with the H100 session and a rerun** |

Against the original `docs/compute.md` plan, the free tiers absorb the entire primary
scaling study. The cost is scheduling: Kaggle's weekly quota and TRC's grant window impose
a calendar that a rented node doesn't.

---

## Reproducibility, since this becomes a paper

Every experiment directory carries:

```
experiments/EN-name/
├── config.yaml      committed BEFORE the run
├── run.py           resumable, idempotent, incremental writes
├── plot.py          regenerates every figure from results/
├── results/         raw outputs, one file per config
├── figures/
└── env.json         auto-captured: platform, chip/GPU model, driver, JAX version,
                     libtpu/CUDA version, commit SHA, wall-clock, cost
```

`env.json` is written by the driver, never by hand. Reconstructing an environment
afterwards is never accurate, and for a paper it has to be exact.

Two rules carried over from `docs/conventions.md` that matter more once this is a
submission:

- **No number in any document without a committed script that regenerates it.**
- **Assert the optimizer trajectory is identical across device counts before comparing
  timings.** Otherwise the scaling figure compares two different computations, which is the
  single most common way scaling results turn out to be wrong.
