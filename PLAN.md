# PLAN.md

> **The gate criteria in `docs/` are authoritative; the one-liners here are navigation.**
> Audited 2026-07-31 and that direction is not a formality: `docs/02` states **six** numbered
> G1 criteria and `docs/03` states **five** for G2, and PLAN summarised each as a sentence.
> Twice the abbreviation dropped the load-bearing part — G1's summary said "1-GPU run" where
> the doc asks for 1-GPU *and* 2-GPU, and one GPU emits no collectives so it cannot test
> sharding at all. Restating a gate is how it drifts. Point at it instead.

## Thesis

Evolution strategies got two credible LLM-scale results in late 2025, using **incompatible
perturbation schemes**, and there is no library that can express both. The incumbent JAX
library (`evosax`) flattens every solution to a single dense vector, which makes
shape-aware perturbation and per-matrix low-rank structure both impossible without surgery
on its base class.

The contribution is a modern sharded ES core where **the perturbation scheme is a
first-class pluggable strategy**, both published algorithms are strategies inside it, and
the population and the rollouts are sharded rather than replicated.

The distribution state is replicated. That was originally claimed as sharded; `docs/02` C1.4
records why it is not, and why replicating it is correct rather than a compromise.

~~Secondary, conditional: low-rank perturbation may be the first ES regime at LLM scale
where population size *exceeds* sampling dimension, which is where coupling / quasi-Monte
Carlo sample design stops being rounding error.~~ **Tested and answered "no", 2026-07-30.**
Rank 1 does reach `N/d_eff = 42.7`, so the regime is real; sample design does not help there.
Measured cosine tracks `√(N/d_ambient)` and is blind to the design. Phase 3 is dropped.
`docs/01`, "The answer: no".

---

## Phases

| # | Phase | Output | Compute | Est. | Gate |
|---|---|---|---|---|---|
| 0 | Estimator harness | Plot + API requirements + go/no-go | 1 GPU, ~1 day | 1–2 wk | G0 |
| 1 | Sharded core | The library | CPU (8 fake devices) + 1–2 GPU | 6–10 wk | G1 |
| 2 | Scaling benchmarks | The headline numbers | 8 GPU, 4–6 h | 2–3 wk | G2 |
| ~~3~~ | ~~Coupling at scale~~ | **DROPPED, G0 = no** | — | *3–4 wk reclaimed* | — |
| 4 | Paper | Submission + artifact | — | 4–6 wk | — |

Total: roughly 3–4 months to a usable library, ~6 months to a submission — less Phase 3's
3–4 weeks, reclaimed when G0 came back negative.

**The benchmark campaign and the paper it feeds are specified separately**, because the
free-tier compute imposes its own calendar: `docs/05-paper.md` (claims → experiments →
figures) and `docs/06-benchmark-runbook.md` (Kaggle, TRC, GCP mechanics). Read those
before scheduling Phase 2 — Kaggle's TPU v5e-8 absorbs most of it at zero cost, and TRC's
grant window has a timing trap that costs you the free compute if you apply too early.

---

### Phase 0 — Estimator harness → `docs/01-phase0-estimator-harness.md`

Measure the *statistical* efficiency of ES gradient estimators against an exact oracle,
on one GPU, before building any infrastructure.

Why first: the result determines the library's central abstraction. If coupling matters
under low-rank perturbation, "perturbation strategy" is a first-class pluggable component
and that's the API. If it doesn't, the library is simpler and shouldn't carry a
speculative abstraction for three months. That is the one decision that's expensive to
reverse later.

It is also not a detour — it requires a batched low-rank forward pass and a
fitness-evaluation loop, which is a vertical slice of the library, written once and kept.

**Gate G0**: do the rank-1 estimator-quality curves separate across sampling schemes at
`N/d_eff ≳ 1`, when full-rank curves at `N/d_eff ≪ 1` do not?
- **Yes** → strategy abstraction is load-bearing; Phase 3 is live.
- **No** → strategy abstraction still needed (two algorithms, two schemes) but stays thin;
  drop Phase 3, reclaim a month.
- **Ambiguous** → record it, proceed to Phase 1, revisit after Phase 2 with real tasks.

> **ANSWERED 2026-07-30: no.** No separation at any rank, sigma or population, out to
> `N/d_eff = 42.7`, with the treatment verified maximal (an exactly orthonormal 512-member
> design against i.i.d.). Measured cosine tracks `√(N/d_ambient)` and is blind to sample
> design. 456 configs, `R = 30`, 13.07 h on the RTX 3080.
>
> **Phase 3 is dropped and the month is reclaimed.** The abstraction stays and stays thin.
> Full answer: `docs/01-phase0-estimator-harness.md` → "The answer: no".
> Open questions that survive: `docs/BACKLOG.md`.

---

### Phase 1 — Sharded core → `docs/02-phase1-sharded-core.md`

The library. `Mesh` / `NamedSharding` / `shard_map`, pytree-native ask/tell, both
algorithms as strategies, replicated distribution state with a per-coordinate diagonal
(C1.4), seed derivation from member index.

**Gate G1**: 8-fake-device CPU run and a **2-GPU** run produce matching updates for a fixed
seed; both published algorithms run end-to-end on a small task; communication volume in the
update path is measured and matches the analysis.

> **Said 1-GPU until 2026-07-31, and that was an abbreviation that dropped the point.**
> `docs/02` asks for a 1-GPU *and* a 2-GPU run. One GPU emits no collectives at all, so it
> checks numerics and cannot check sharding — which is the entire reason the criterion exists.
> Corrected here rather than satisfied on the easier reading.
>
> **Status 2026-07-31: capabilities C1.1–C1.7 complete, 5 of 6 criteria met.** Outstanding:
> the 2-GPU run (`docs/06` T2′). The 1-GPU half passes on the RTX 3080.
>
> Criterion 1's "under two minutes" also predates the tier split and contradicts
> `docs/conventions.md`, which measures and budgets ~2 min fast / ~6 min full. One of the two
> needs to win; the suite currently runs 149 s and 363 s.

---

### Phase 2 — Scaling benchmarks → `docs/03-phase2-benchmarks.md`

Strong and weak scaling across 1/2/4/8 devices. Throughput, wall-clock per generation,
communication volume, memory per device. Compare the two contraction strategies (scalar
all-reduce + replicated regeneration, vs. model-size all-reduce of the partial update)
and find the crossover.

**Gate G2**: `docs/03-phase2-benchmarks.md` states five numbered criteria. The short form is
"a scaling curve worth putting at the top of a README", and the four it omits are the ones that
make the curve trustworthy — the crossover phase diagram, an external comparison at matched
shapes, reproducibility from a committed config, and a limitations paragraph a skeptic would
accept. Read the doc.

---

### Phase 3 — Coupling at scale → `docs/04-phase3-coupling.md` — **DROPPED (G0 = no)**

**Conditional on G0.** Coupled/low-discrepancy sampling, validated end-to-end on task
performance rather than estimator MSE — because lower variance does not straightforwardly mean
better ES (see the smoothing caveat in `docs/00-context.md`).

The schemes themselves (`OrthogonalHD`, `ScrambledSobol`) shipped in Phase 0, since G0 needs
them to answer its own question. What is conditional is making them survive sharding and
validating them on tasks. They turned out to be a noise source handed to a strategy rather than
a wrapper around one; `docs/04` C3.1 records why.

---

## What "done" looks like

A repo a hiring manager can skim in five minutes and come away with:

1. A README whose first figure is a scaling curve across 8 devices.
2. A one-paragraph statement of the architectural claim (no global flattening; perturbation
   as a strategy) with a link to the evosax comparison that motivates it.
3. Both published algorithms runnable from the same API with a two-line diff.
4. A test suite that runs on CPU and includes the device-count-invariance test. **Two tiers,
   not two minutes**: `pytest --fast` is the inner loop (~2 min) and `pytest` is the full
   suite (~6 min). `docs/conventions.md` carries the measured figures and the reasoning; this
   line said "two minutes" until 2026-07-31 and contradicted it.
5. Honest limitations, including anything Phase 0 or 2 disconfirmed.

Nice-to-have, not required: an upstream contribution. The realistic upstream targets are
small evosax PRs (JAX 0.11 modernization, and moving off the deprecated `brax.envs` path)
which can be done in parallel and are independently mergeable.

One correction from having actually installed it: MuJoCo Playground is **not** an alternative
to brax. It depends on `brax>=0.14.2`, which pulls `jaxopt` (last release April 2025, folded
into optax). It is a maintained wrapper *around* brax, not a way off it. Any upstream pitch
that says otherwise is wrong. `docs/02` C1.7.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GCP GPU quota not approved in time | **high** | Request during Phase 0. Mostly de-risked now: the primary scaling study runs free on Kaggle TPU. |
| TRC grant window wasted by applying too early | **high** | Grant is temporary (~30 days) and starts on acceptance. Apply ~week 10. See `docs/06-benchmark-runbook.md` §T3. |
| Kaggle weekly quota throttles the E4 grid | medium | Plan E4 as four ~5-h chunks across two weeks. Resumable driver is mandatory, not optional. |
| Someone publishes sharded ES first | medium | The architectural claim (strategy-pluggable, unflattened) survives even if a scaling result doesn't. Check monthly. |
| evosax maintainer objects to the framing | low-med | This is a standalone library, not a fork. Be complimentary and specific about the flattening issue; offer the modernization PRs regardless. |
| Phase 2 burns budget on a config bug | medium | Full dress rehearsal on 1–2 GPUs with tiny `N` and a hard wall-clock cap. Checkpoint every generation. See `docs/compute.md` §"Not wasting the 6 hours". |
| Estimator MSE turns out to be a bad proxy for task performance | **high — expected** | Stated up front in Phase 0's limitations. G0 gates the *abstraction*, not the algorithmic claim. **Task-level validation was Phase 3's job and Phase 3 is dropped, so it is now `docs/BACKLOG.md` B3 and nothing in this plan closes it.** Say so in the writeup rather than letting the negative read as broader than it is. |
| Scope creep into CMA-ES variants (VD-CMA, LM-CMA) | medium | Out of scope until after G2. They're a good follow-up, not part of the core claim. |
| Scope creep into the ZO variance-reduction literature | medium | Full-rank variance reduction beyond mirrored and `orthogonal_hd` is deferred, deliberately. Control variates, subspace projection, preconditioning and importance mixing are all plausible and none is what this project is asking. Sample design under low rank **was** the question and G0 answered it negatively, so the "revisit after G2" no longer has a live claim attached: full-rank variance reduction is now simply out of scope rather than deferred behind a pending result. |

---

## Open questions to resolve as you go

1. **Contraction strategy.** Scalar all-reduce + replicated seed-regeneration, versus
   all-reducing the model-sized partial update. Both are defensible; the crossover depends
   on `N`, `d`, and device count. **The volume half is measured** (Phase 1,
   `experiments/phase1/comms.py`): A moves `8N` bytes, B moves `4d`, all 27 rows agree with
   the prediction to the byte, and the crossover sits at `N = d/2`, moving with the model
   rather than the device count. **The time half is still Phase 2's**, and it is the one the
   claim is about — bytes are not latency and simulated devices model no interconnect. Do not
   assert "ES only needs a scalar all-reduce" in any public writeup until that exists; it is
   true only for the first strategy.
2. **Embedding layers under low-rank perturbation.** EGGROLL's reference implementation
   raises `NotImplementedError` for the embedding path. Unsolved; a real contribution if
   cracked; explicitly *not* required for any gate.
3. ~~**Where the sharded distribution state actually pays.**~~ **Answered 2026-07-31: it
   does not, here.** Sharding an `O(d)` state on a population-parallel mesh costs a gather
   every generation to save memory already spent replicating the model, and a CMA-family
   strategy would need a protocol change that Gate G0 argues against. Isotropic ships, with a
   per-coordinate diagonal that needs no protocol change. `docs/02` C1.4.
4. **Fitness shaping is a synchronization point, and it is not only centered ranks.**
   Centered ranks need a global sort; `centered` needs the global mean; `group_relative`
   reduces over the member axis, which is the sharded one. All three are a barrier and only
   `none` is not. Cheap in bytes, not free in latency. Measure the cost in Phase 2 (E10).
