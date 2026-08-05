# A correctness check that failed on correct code

**Status: draft, written by Claude Code for Andres to rewrite.** Ground rule 1 in
`CLAUDE.md` says anything that ends up in a README or a talk gets drafted here and written
for real by him. The measurements are real and cited; the prose is a starting point.

This is a postmortem of a bug hunt in `shardes`, a library for running evolution strategies
across many GPUs. The bug turned out not to be in the library. It was in the thing checking
the library, and the root cause was a property of GPU floating-point arithmetic that does
not exist on a CPU.

It is written for someone who has not seen this project and who is early in learning how
CPUs and GPUs differ numerically. Everything you need is in section 1.

---

## 1. Background

### 1.1 What the library does

Evolution strategies (ES) optimise a model without gradients. One step looks like this:

1. Take the current parameters.
2. Generate `N` random perturbations of them (`N` is the *population*, here 64 to 1024).
3. Score each perturbed copy on some task. Each score is one number.
4. Combine the scores into a single update and apply it.

Step 3 is the expensive part, and each of the `N` members is independent, so you can put
different members on different devices. That is what `shardes` does: it spreads the
population across GPUs.

### 1.2 The property under test

If you fix the random seed, **running on 1 device and on 8 devices must produce the same
update**. Nothing about the answer should depend on how many machines you split the work
over. The project calls this *device-count invariance*, and `CLAUDE.md` calls the test for
it "the most important test in the repo".

There are two ways to do step 4, and they are held to different standards:

- **Strategy A** gathers the `N` scores onto every device, then every device regenerates all
  `N` perturbations and combines them in the same order. Every device does identical
  arithmetic, so the answer must be **bitwise identical** no matter how many devices there
  are.
- **Strategy B** has each device combine only its own share, then adds the partial results
  together across devices. The order of that final addition depends on how many devices
  there are, so B is allowed to differ in the last digit. Tolerance: `1e-5` relative.

Holding B to bitwise equality would fail forever. Holding A to a tolerance would hide real
bugs. Getting that split right is the whole point of the check.

### 1.3 Why the check exists at all

The library's headline output is a scaling curve: how much faster does this get on 8 devices
than on 1? A speedup number is only meaningful if both runs did the *same computation*. So
before any timing number is quoted, a guard script re-runs one step at each device count and
asserts the results match.

That guard is what failed.

### 1.4 Floating point addition is not associative

This is the one piece of numerical background that matters, and it is worth seeing rather
than being told.

`(a + b) + c` and `a + (b + c)` can give different answers in floating point, because each
addition rounds to the nearest representable number and the rounding depends on the
magnitudes involved. Adding 1.0 and ten copies of 1e-7 in float32:

```
eps(float32) = 1.192e-07
  small values added first: 1.000000954
  large value added first : 1.000001192
  identical: False
```

Same eleven numbers, two orders, two different answers. The small values accumulate into
something big enough to survive rounding if you add them together first, and get swallowed
one at a time if you add them to 1.0 first.

At realistic sizes the effect persists. Summing 65,536 random float32 values sequentially
versus in a balanced tree (what NumPy does) gives a relative difference of **2.77e-06**.

Neither answer is wrong. They are different roundings of the same exact sum.

### 1.5 Why this bites GPUs and not CPUs

Adding up `N` numbers is a *reduction*. On a CPU it is usually a loop, in a fixed order, and
you get the same answer every time by accident rather than by design.

A GPU has thousands of threads and cannot use a simple loop. It splits the array, has each
thread sum a chunk, then combines the chunks, often in a tree, sometimes using atomic
operations that complete in whatever order the hardware schedules them. **The compiler picks
which scheme to use, and its choice depends on the shape of the data.** Modern compilers also
*autotune*: they try several implementations, time them, and keep the fastest.

So:

- A CPU reduction is deterministic mostly by luck.
- A GPU reduction is deterministic only if you ask for it.

XLA, the compiler JAX uses, has a flag for asking: `--xla_gpu_deterministic_ops=true`.

---

## 2. The symptom

### 2.1 What a "configuration" is

The benchmark sweeps a grid, and one point in that grid is a *configuration*: one complete
run of the library with one setting of every knob. The calibration sweep had six axes:

| axis | values | count |
|---|---|---|
| scaling mode | strong (fix the total population, add devices) | 1 |
| device count `D` | 1, 2 | 2 |
| model width `d_model` | 256, 512 | 2 |
| population `N` | 64, 256 | 2 |
| perturbation strategy | `iid_gaussian`, `seed_regenerated`, `mirrored_lr1`, `lowrank_r1` | 4 |
| contraction strategy `how` | A, B | 2 |

`1 x 2 x 2 x 2 x 4 x 2 = 64`. One of them is named
`mode=strong__D=1__d=256__N=64__s=iid_gaussian__how=A`.

Two things this is easy to misread:

- **A configuration is not a perturbation.** Each configuration runs 9 optimisation steps (3
  to warm up, 5 timed, 1 for the guard), and each step generates `N` perturbations. The
  failing configuration used 64 perturbations per step. `N` is the population axis; 64 is
  also the number of configurations, which is an unhelpful coincidence.
- **`d_model` is not a parameter count.** It is the width of the model being optimised: a
  single transformer block with six square weight matrices (`wq, wk, wv, wo, w_up, w_down`),
  one head, no learnable norm scale.

  | `d_model` | matrices | parameters | size in float32 |
  |---|---|---|---|
  | 256 | six 256x256 | 393,216 | 1.6 MB |
  | 512 | six 512x512 | 1,572,864 | 6.3 MB |

  The block sets `d_ff = d_model`, where a real transformer uses 4x, so that every matrix
  has the same effective dimension. A single block is not an LLM; it is a stand-in with the
  right shape characteristics.

### 2.2 The failure

The guard ran on a Kaggle node with two Tesla T4 GPUs. Sixty-three configurations passed.
One failed:

```
    d      N strategy      how    D    exact  max rel dev
  256     64 lowrank_r1    A     1,2   False  6.32e-03
```

Read that as: on a 393,216-parameter model, with 64 perturbations per step, using the
`lowrank_r1` perturbation scheme and contraction strategy A, the update computed on 1 GPU
differed from the update computed on 2 GPUs by 6.32e-03 relative. Strategy A is the one that
is supposed to be bitwise identical.

Four things made this look like a genuine bug in the library:

1. **It was large.** 6.32e-03, not 1e-7. Last-digit rounding noise is around 1e-7 here. This
   was four orders of magnitude bigger.
2. **It was reproducible.** Re-running gave the same number.
3. **It was specific.** `d=512, N=64` was fine. `d=256, N=256` was fine. Only this one cell.
4. **It was strategy-specific.** The other three strategies were exactly 0.

Every one of those signals normally points away from "numerical noise" and towards "logic
error". All four were misleading.

---

## 3. Clearing the ground: two of the failures were not real

Before this investigation, the guard reported four failures, not one. It measured error as
`max(|a - b| / |b|)` computed element by element. That metric divides by individual
components, and a random projection of a parameter vector has components near zero all the
time, so a tiny absolute difference next to a near-zero component produces a huge ratio.

Replacing it with a relative error in the L2 norm, which asks "is this the same vector",
cleared two of the four:

| configuration | old metric | corrected metric |
|---|---|---|
| `seed_regenerated/B` | 1.11e-05 (fail) | 1.51e-07 (pass) |
| `lowrank_r1/B` at `d=512, N=256` | 3.80e-05 (fail) | 1.89e-07 (pass) |
| **`lowrank_r1/A` at `d=256, N=64`** | **1.15e-01** | **6.32e-03 (still fails)** |

Worth stating plainly: a bad error metric had previously caused a TPU failure to be blamed on
`bfloat16` arithmetic, and that explanation was written into three documents as measured fact
before anyone checked the metric. **Before debugging a number, check that the number measures
what you think it does.**

---

## 4. First hypothesis, and a wrong turn worth studying

### 4.1 The reproduction that was not one

The obvious first move is to reproduce the failure somewhere cheap. I ran the same
configuration on the development laptop using 8 simulated CPU devices, and got **6.33e-03**,
against the GPU's 6.32e-03.

That looks like a clean reproduction. It is not. It is the most instructive mistake in this
whole investigation.

The benchmark driver is *resumable*: it writes one result file per configuration and skips
configurations already on disk, so a killed run can continue. My two commands were:

```
# first, with no environment override: the laptop has one GPU
python run.py --config repro.yaml        # ran D=1 on the GPU, skipped D=2 ("needs more devices")

# second, forcing 8 simulated CPU devices
JAX_PLATFORMS=cpu ... python run.py --config repro.yaml   # ran D=2 on CPU, SKIPPED D=1
```

The `D=1` result came from a GPU. The `D=2` result came from a CPU. The resume feature had
silently combined them, and the guard compared them without noticing. The number I got was
the difference between two *backends*, not between two device counts.

The matching digits, 6.32 versus 6.33, felt like strong confirmation. They were coincidence.
Getting the answer you expected is not evidence that you measured the right thing.

Re-running both device counts in one command on one backend:

```
  256   64 lowrank_r1  A  1,2  True   0.00e+00
  256   64 lowrank_r1  B  1,2  False  7.85e-08
```

Bitwise identical on CPU. **The failure was GPU-only.**

### 4.2 The fix that came out of the wrong turn

The guard's docstring said the comparison assumed one platform. It was a comment, not a
check. A resumable harness will eventually mix environments, so the guard now records the
backend, chip model, JAX version, commit and compiler flags with every result and refuses to
compare across them:

```
256  64  lowrank_r1  A  1,2   -   MIXED ENV
FAIL: ... spans environments (device_platform: cpu vs gpu; device_kind: RTX 3080 vs cpu)
```

A guard that can be fooled by its own resume feature is not a guard.

---

## 5. Second hypothesis: sharded scores plus a discontinuous transform

### 5.1 The reasoning

Under strategy A, every device regenerates all `N` perturbations and combines them with the
same fixed operation, so **the combining step cannot depend on the device count**. Working
backwards, the only input to that step that *is* split across devices is the vector of
scores: each device scores `N/D` members and the results are gathered.

Scoring 64 members in one batch and scoring 32 members in each of two batches are different
shapes. From section 1.5, different shapes are exactly what lets a GPU compiler choose
different reduction algorithms. So the scores were allowed to differ in the last digit.

On its own that would produce a 1e-7 difference, not 6e-3. But the scores are not used
directly. They pass through *fitness shaping*, and the default is `centered_ranks`: sort the
members by score and replace each score by its rank. **Sorting is discontinuous.** If two
members are nearly tied and a last-digit difference flips their order, the update changes by
a whole rank step, not by a rounding error.

That predicted the exact signature observed: the update's *magnitude* barely moved (norms
agreed to 3e-5) while its *direction* moved a lot (6e-3).

### 5.2 A promising detail from the rehearsal

Before running anything on rented hardware I rehearsed the diagnostic on CPU. It reported
something I had not thought to look for:

| strategy | distinct score values (of 64) | smallest gap between adjacent scores |
|---|---|---|
| `lowrank_r1` | **63** | 0.0 |
| `mirrored_lr1` | ties present | 0.0 |
| `iid_gaussian` | 64 | 4.77e-06 |

The low-rank strategies had **exactly tied scores**. That makes sense: a rank-1 perturbation
at `sigma=0.01` changes a `d=256` loss by less than one float32 ulp, so members collide
exactly. And ties are worse than near-ties, because then the sort's tie-breaking rule alone
decides the order, and nothing requires a GPU to break ties the same way at two different
shapes.

Better still, the split matched the bug: ties in the low-rank strategies, none in
`iid_gaussian`, and the failure was in a low-rank strategy.

### 5.3 The hypothesis appears to die

The diagnostic ran on the actual 2x T4. Every link in the chain looked broken:

| measurement (`lowrank_r1/A`, `d=256`, `N=64`) | result |
|---|---|
| raw scores bitwise equal across device counts | **True** (0 of 64 differ) |
| distinct score values | **64 of 64**, no ties at all on GPU |
| members whose rank changed | **0 of 64** |
| shaped weights, relative difference | 0.000e+00 |
| **resulting parameters after one step** | **0.000e+00** |

The scores were identical, nothing was reordered, and one clean step produced bitwise
identical parameters at both device counts. I recorded the hypothesis as refuted and moved
the search elsewhere.

**That was a mistake, and it is the second most instructive error in this investigation.**

Look at the last row. The parameters were identical, which means *this process did not
exhibit the bug at all*. Every other row in the table is downstream of that: of course no
rank changed, because nothing differed anywhere. A null result in a run where the failure is
absent says nothing about a run where it is present.

The right reading was "this process did not reproduce the failure, so this measurement is
uninformative". The reading I took was "the scores are identical, therefore scores are not
the mechanism". The evidence could not support that, and it sent the investigation somewhere
else for no reason.

### 5.4 The observation that should have been the clue

The same run also confirmed the failure was still there: the driver reported 6.32e-03 for
that cell, on that node, at that commit, in the same session where a single step reported
0.000e+00.

Same hardware, same code, same configuration. One step: identical. The driver: not identical.

I read this as "the driver does something a single step does not", and moved the search into
the harness. That turned out to find the real trigger (section 6), so the detour was not
wasted. But there was a simpler reading available: **the same computation, run twice on the
same node, gave two different answers.** Not two device counts. Two runs. That is the
signature of an arithmetic choice being made per process, and it was visible here, one
section earlier than I noticed it.

---

## 6. The decisive experiment

Two candidates remained:

1. **It is not reproducible at all.** GPU nondeterminism, which the guard would then be
   measuring instead of the library. The project's own GPU test cell had always set
   `--xla_gpu_deterministic_ops=true`; the benchmark kernels never had.
2. **The warm-up changes the answer.** The driver compiles the step function during its
   timing loop and reuses that compiled version for the guard, so anything decided during
   warm-up is baked in.

The probe tested both, and ran the whole thing twice: once as the benchmark actually ran, and
once with the determinism flag set.

```
PASS 1 - no flag (how the sweep ran)
  step 1  D=1 twice: identical      D=2 twice: identical
  step 2  warmups=0: D1 vs D2  8.93e-03     warmups=8: D1 vs D2  8.93e-03
          warmed vs cold, same D: 0.00e+00
  step 3  measure() twice, same D: identical
          measure() D=1 vs D=2: 6.32e-03

PASS 2 - with --xla_gpu_deterministic_ops=true
  every number: 0.000e+00
```

Reading it:

- **The warm-up hypothesis is dead.** Warmed and cold give identical answers, and the
  divergence is already present with zero warm-up.
- **It is not run-to-run noise.** Every repeat inside a process is bitwise identical, which
  is precisely why it looked like a logic bug.
- **The flag fixes it completely.**

(The 8.93e-03 and 6.32e-03 are the same underlying difference measured two ways: one is the
raw parameter vector, the other a 16-dimensional random projection of it that the guard
records.)

### 6.1 The observation that pins the mechanism

There is one more result, and it is the strongest of the lot. The *earlier* diagnostic
measured that exact quantity, one step, `lowrank_r1/A`, no flag, as **0.000e+00**. This probe
measured it as **8.93e-03**. Same node type, same commit, same code path, both without the
flag.

**Two processes, two different answers.** Combined with "every repeat within a process is
identical", that is the fingerprint of a choice made once per process and then cached, which
is exactly how autotuning behaves: it times candidate kernels and keeps the winner, and
timings vary between processes.

---

## 7. Root cause

There are two halves, and conflating them is what made the investigation take as long as it
did. One is the *trigger*: what perturbs the computation. The other is the *amplifier*: why a
perturbation that should be invisible turned into 6.32e-03.

### 7.1 The trigger

Without `--xla_gpu_deterministic_ops=true`, XLA:GPU selects reduction algorithms per shape,
and tunes that selection by measured kernel timing.

A reduction over 64 members and a reduction over 32 members are different shapes, so they can
get different algorithms, with different summation orders. By section 1.4, different
summation orders give different roundings of the same exact value. The scores therefore move
by roughly one ulp.

| observation | explanation |
|---|---|
| repeats within a process are bitwise identical | the choice is made once and cached |
| `D=1` and `D=2` disagree | different shapes, different algorithms |
| two processes disagree with each other | autotuning picks by measured time |
| CPU is unaffected | CPU reductions use a fixed order |
| the flag makes it exactly zero | the flag removes the choice |

### 7.2 The amplifier

One ulp is 1e-7 relative. The failure was 6.32e-03, four orders larger. Something multiplied
it, and section 5 named the candidate correctly before I discarded it on bad evidence.

Measured directly, on CPU where the arithmetic is fixed, by taking the two closest members
and exchanging them:

| strategy | shaping | input change | update change | amplification |
|---|---|---|---|---|
| `lowrank_r1` | `centered_ranks` | 3.37e-08 | **8.93e-03** | **265,000x** |
| `lowrank_r1` | `centered` (continuous) | 3.37e-08 | 8.44e-07 | 25x |
| `iid_gaussian` | `centered_ranks` | 3.54e-07 | 9.06e-03 | 25,600x |

**8.93e-03 is what the unflagged `D=1` versus `D=2` comparison measured, to three
significant figures.** The divergence was one rank swap.

The remaining question is why *this* configuration. Because its members were unusually close
together:

| strategy | `d` | closest pair |
|---|---|---|
| `lowrank_r1` | **256** | **2.00 ulp** |
| `iid_gaussian` | 256 | 21 ulp |
| `lowrank_r1` | 512 | 63 ulp |

At 2 ulp, a one-ulp perturbation is enough to reorder the pair. At 21 or 63 it is not. That
single number explains every piece of shape- and strategy-specificity that made this look
like a logic bug: `d=512` always passed because it had 63 ulp of headroom, and `lowrank_r1`
failed where `iid_gaussian` did not because rank-1 perturbations produce more tightly
clustered scores.

### 7.3 The full chain

```
different device count
  -> different batch shape for scoring
  -> different reduction algorithm            (removed by the determinism flag)
  -> scores move by ~1 ulp
  -> two members 2 ulp apart swap order       (a property of the configuration)
  -> centered_ranks moves a whole rank step   (a property of rank shaping)
  -> update moves by 8.93e-03                 (265,000x the input change)
```

**The perturbation strategy was never at fault.** `lowrank_r1` contributed only tightly
clustered scores. Any strategy is one rank swap away from a 1e-2 update change: `iid_gaussian`
measures 9.06e-03 for the same experiment.

On evidence: the amplification, the gap in ulp, and the flag fixing the failure are all
measured. That XLA's algorithm selection specifically is the trigger is inference, supported
by those measurements and by XLA's documented behaviour, but not observed at the level of
emitted kernels.

---

## 8. Resolution

The fix removes the trigger. It does not touch the amplifier, and section 8.2 is about why
that distinction matters more than it first appears.

The fix is configuration, not code:

1. **The driver refuses to run.** On a GPU without the flag it exits with an error naming the
   flag, rather than producing a number nobody should trust. It cannot set the flag itself:
   XLA reads `XLA_FLAGS` once when the GPU backend starts, which is before the driver's code
   runs.
2. **The guard treats compiler flags as part of the environment.** A run with the flag and a
   run without it are as incomparable as a CPU run and a GPU run.
3. **The benchmark kernels set it.**

Verification on 2x T4, all 64 configurations:

```
OK: 32 strong-scaling groups, every device count ran the same thing.
```

`lowrank_r1/A` is now exactly `0.00e+00` where it was `6.32e-03`. Every strategy A is bitwise
identical across device counts. Strategy B runs from 7.33e-08 to 4.16e-07 against its 1e-5
tolerance, with one configuration landing on exactly 0.00e+00, which is the expected
last-digit behaviour described in section 1.2: B is *allowed* to differ, not required to.

### 8.1 What determinism cost

This is the part most worth measuring rather than assuming, because "deterministic mode is
slow" is received wisdom.

| | compute | total wall | overhead |
|---|---|---|---|
| no flag | ~23 s | 1302 s | 20.0 s/config |
| deterministic | ~23 s | 1932 s | 29.8 s/config |

Per-step compute is **unchanged**: the median ratio across 64 matched configurations is
**1.023**, with outliers scattering from 0.70 to 1.36 in both directions, which is ordinary
timing noise on a shared node rather than a systematic penalty.

The entire increase is **compile time**, about +9.8 s per configuration. Across the full
256-configuration benchmark that is roughly 45 extra minutes.

So in this workload the correctness fix is free in arithmetic throughput and costs
compilation. That will not generalise to every workload. Deterministic reductions can be
genuinely slower when a kernel relies on atomics for speed. The point is that it was cheap
*here*, and that was established by measuring rather than by assuming in either direction.

A related detail from the same table: at these shapes, compute is **1.8% of wall time**. This
benchmark is mostly measuring the compiler.

### 8.2 The fix removes the trigger, not the sensitivity

Setting the flag makes `D=1` and `D=2` run identical arithmetic, so the zero is structural
rather than lucky: under strategy A every device regenerates all `N` members and contracts
them in the same order, and the per-member scores were measured bitwise equal. It is not a
coincidence that the deterministic algorithm happens to agree with itself.

But the 265,000x amplification is untouched. Anything else that perturbs the scores by an ulp
re-triggers the whole chain: a different GPU model, a newer XLA or cuBLAS, TF32 instead of
fp32, a different batch or sequence length. The flag makes the benchmark **reproducible**,
not **well conditioned**, and those are different claims.

That prompted a check for the second property, `experiments/phase2/noisefloor.py`, which
reports how far apart the closest two members are in ulp and fails a configuration whose
closest pair sits inside 16 of them. Running it against the real sweep is uncomfortable:

| population `N` | 32 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| adjacent pairs within 16 ulp | 0 | 4 | 18 | 65 | **242 of 1023** |

**19 of 24** configurations at `d=512` fail, and at `N=1024` roughly a quarter of adjacent
pairs are inside the noise floor, including 5 to 9 pairs that are exactly equal.

The reason is arithmetic rather than anything specific to this library. Separating `N`
members by `m` ulp needs a relative spread of scores of about `N * m * eps`, which is 5e-4 at
`N=256, m=16`. Large populations are the hard case, and ES wants large populations.

Raising `sigma` is the obvious remedy and it only half works. Measured:

| configuration | sigma 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|
| `lowrank_r1`, `d=256, N=64` | 0 ulp | 56 | 2555 | 4742 |
| `iid_gaussian`, `d=256, N=256` | 1 ulp | 8 | 6 | 2 |

It rescues small populations and does nothing for large ones, because a larger `sigma` raises
the loss magnitude too and takes the ulp up with it.

**Exact ties used to be the worse case, and were fixed separately.** `centered_ranks`
originally gave tied members *different* weights, chosen by the sort's tie-break, so a pair
that tied on one backend and differed by an ulp on another could order either way. It now
gives them the average of the ranks they span, so an exact tie is no longer a source of
disagreement. Appendix B has the implementation and what it cost.

That fixed one failure mode and left the larger one alone: members one ulp apart still get
distinct ranks and can still swap, which is 242 of the 1023 adjacent pairs at `N=1024`
against 5 to 9 exact ties. **The `close` column in the table above is the number to read,
not `ties`.**

It also turned out to matter far more in Phase 0 than in Phase 2. At `d=512, N=16384,
sigma=1e-3` two thirds of members share a fitness exactly, and at `N=262144` it is 96%,
because a perturbation that small moves the loss by less than one float32 ulp. Every one of
Phase 0's 228 committed rank-shaped results contained ties and had to be regenerated. The
numbers barely moved (median absolute change in cosine 2.35e-07 against a typical 1.02e-02)
and relative MSE improved in 218 of 228, which is the expected direction: equal weights for
tied members add less noise than weights ordered by member index.

The measured alternative that does address near-ties is continuous shaping: `centered` sits
at 25x amplification against `centered_ranks`'s 265,000x, four orders of magnitude better.
It is a different optimizer, though. Rank shaping exists to buy robustness to outliers and to
reward scale, and giving that up to gain reproducibility is a real trade rather than a free
improvement.

### 8.3 What this does and does not invalidate

Phase 2 measures wall-clock scaling. Timing does not depend on how members are ranked, so the
strong-scaling, weak-scaling and crossover results stand.

What is narrower than it sounds is the invariance claim. The guard verifies that the same
program under fixed arithmetic produces the same update at any device count. It does not
establish that the optimizer is device-count invariant in a way that survives a change of
hardware, and at `N=1024` it is not: the update there is one rounding decision away from
moving by 1e-2.

Both halves belong in the limitations section, and appendix A drafts them.

---

## 9. What generalises

**Reproducible does not mean correct.** The strongest reason this looked like a logic bug is
that it reproduced exactly. On a GPU, a decision cached once per process produces perfectly
repeatable wrong comparisons. Repeat the measurement in a *fresh process* before concluding
that a stable number is a real one.

**"Deterministic" is at least three separate claims.** Same answer twice in one process; same
answer for different shapes; same answer across processes. Hardware and compilers give the
first for free and the others only when asked. This bug satisfied the first and violated the
other two.

**Magnitude does not distinguish rounding from logic.** A 6e-3 difference looks far too large
to be a last-digit effect. It was a last-digit effect that had passed through the rest of the
computation. Ask what the pipeline does to a small perturbation before ruling out rounding on
size alone.

**Check that your metric measures what you think.** A componentwise relative error turned
1.89e-07 into 3.80e-05 and sent an earlier investigation chasing `bfloat16`.

**Features interact with checks.** Resume plus a guard that assumed one environment produced
a confident, completely fabricated reproduction. Any harness that skips work it thinks is
done should record the conditions that work was done under.

**Test what you benchmark.** The strategy that failed was in the benchmark configuration but
in neither the invariance test nor the rehearsal. Three safety nets, and the same strategy
fell through all three, because a related strategy wrapped it and looked like coverage.

**A null result is not a refutation.** The scores-plus-sorting hypothesis was correct, and I
discarded it on a measurement taken in a process where the bug was not present. Every row of
that table read "no difference" because the *last* row read "no difference": nothing had
diverged, so of course nothing was reordered. Before treating a measurement as evidence
against a mechanism, check that the run you measured actually exhibited the thing you are
explaining.

**Separate the trigger from the amplifier.** Two questions hid inside one symptom: what
perturbs the computation, and why a 1e-7 perturbation becomes 1e-2. Fixing the trigger made
the failure disappear, which is exactly the kind of success that stops an investigation one
step early. The amplifier is the more interesting half and would have gone unexamined.

**A fix that makes the symptom vanish may not have made the system robust.** Pinning the
arithmetic gives bitwise agreement because it forces both sides to run the same instructions,
not because the computation became stable. The distinction is invisible while the pin holds
and reappears on the next machine.

---

## 10. Reproducing this

The two diagnostics are committed so the reasoning can be re-run rather than taken on trust:

- `experiments/phase2/kaggle/lrdiag/` tests the scores-and-sorting hypothesis. Reports
  whether raw scores differ across device counts, how many members change rank, and how many
  share a score exactly.
- `experiments/phase2/kaggle/nondet/` tests reproducibility and warm-up, with and without the
  determinism flag.

Both run on a free Kaggle 2x T4 node in about 15 minutes.

---

## Appendix A: draft limitations paragraph (G2 criterion 5)

`docs/03` asks for "a limitations paragraph a sceptic would accept". Draft, for Andres to
rewrite:

> **Reproducibility of the update.** The scaling results in this section were produced with
> `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, and the driver refuses to run on a GPU
> without it. This is load-bearing rather than hygienic. Without it XLA selects reduction
> algorithms per shape, so scoring `N` members on one device and `N/D` on each of `D`
> devices can use different summation orders and the scores move by about one ulp. Fitness
> shaping is a rank transform, which is discontinuous: exchanging the two closest members
> moves the update by 8.9e-03 against a 3.4e-08 change in the scores, an amplification of
> 2.7e5, measured on a 2x T4. The trajectory guard consequently verifies a specific claim:
> the same program, on the same hardware, with the same compiler flags, produces the same
> update at every device count. It does not establish that the update is stable under a
> change of GPU, of XLA version, or of matmul precision, and at large populations it is not.
>
> **Score separation.** `experiments/phase2/noisefloor.py` reports how far apart the closest
> two members are, in units of the last place of the loss. At `d=512` the closest pair is 37
> ulp apart at `N=32` and exactly zero at `N=512` and above, with 242 of 1023 adjacent pairs
> inside 16 ulp at `N=1024`. Separating `N` members by `m` ulp needs a relative spread of
> scores of about `N * m * eps`, so this is a property of float32 against large populations
> rather than of any strategy here: `iid_gaussian` and the low-rank strategies both hit it.
> Where members are within the noise floor their relative ranking is decided by rounding.
> The resulting update is still a valid ES update, since the members concerned are
> statistically indistinguishable, but it is not reproducible across a change of arithmetic.
> The timing results are unaffected: wall clock does not depend on the ranking.
>
> **What we did not do.** We did not re-run the sweep in float64, which would raise the
> resolution but changes the arithmetic being benchmarked. We did not adopt continuous
> shaping, which measures 25x amplification against the rank transform's 2.7e5 but is a
> different optimizer. Both are recorded here as the honest alternatives rather than
> presented as future work.

## Appendix B: midrank shaping, as shipped

**Shipped, having started here as a proposal.** Shaping is estimator math, which `CLAUDE.md`
ground rule 1 reserves for Andres, so this appendix drafted it with its reasoning and its
limits and he took it from there. Kept as the record of why, and of what it does not buy.

`centered_ranks` before the change:

```python
ranks = jnp.argsort(jnp.argsort(fitness)).astype(fitness.dtype)
return ranks / (n - 1) - 0.5
```

Two members with *exactly equal* scores receive *different* weights, because `argsort` breaks
the tie by index. At `n=4` with scores `[3, 1, 1, 2]` the tied pair gets `-0.5` and `-0.167`,
a third of the full weight range apart. Which member gets which is decided by the sort, so a
pair that ties on one backend and differs by an ulp on another can order either way.

Midrank gives tied members the average of the ranks they span, which is the standard
statistical treatment of ties and makes the weight vector a function of the score *multiset*
rather than of the sort's tie-break. Sketch, `O(n log n)`, jittable with static shapes:

```python
def centered_ranks(fitness):
    n = fitness.shape[0]
    if n < 2:
        return jnp.zeros_like(fitness)
    order = jnp.argsort(fitness)
    s = fitness[order]
    # A new tie group starts wherever the sorted value changes.
    starts = jnp.concatenate([jnp.array([True]), s[1:] != s[:-1]])
    group = jnp.cumsum(starts) - 1
    positions = jnp.arange(n, dtype=fitness.dtype)
    first = jax.ops.segment_min(positions, group, num_segments=n)
    count = jax.ops.segment_sum(jnp.ones_like(positions), group, num_segments=n)
    mid = (first + (count - 1) / 2)[group]          # average rank within each tie group
    ranks = jnp.zeros_like(fitness).at[order].set(mid)
    return ranks / (n - 1) - 0.5
```

**`first + (count - 1) / 2`, not `sum(positions) / count`.** The two are equal in exact
arithmetic and not in float32: summing positions reaches `n^2 / 2`, which passes 2^24 and
stops being exactly representable. With every member tied, the sum-based form is off by 0.5
ranks at `n = 2^16` and by 1.4 at `n = 2^18`, which is the population `lowrank.py` cites.
Taking the group's first position keeps every intermediate below `n`.

Checked before proposing, on the `[3, 1, 1, 2]` example above:

```
centered_ranks  : [ 0.5     -0.5     -0.16667  0.16667]
centered_midrank: [ 0.5     -0.33333 -0.33333  0.16667]

tied members now get the SAME weight
no ties -> identical to centered_ranks: True
survives jit: True
weights invariant to exchanging the tied pair: True
```

The second line is the one that matters for adopting it: with no ties present it reproduces
the current shaping exactly, so it is not a change in behaviour anywhere except where the
current behaviour is arbitrary.

**What it buys, stated honestly.** It removes exactly one failure mode: exact ties. At
`d=512, N=1024` the noise floor check counts 5 to 9 exact ties and 242 pairs within 16 ulp,
so this addresses a few percent of the configurations at risk. Members that are one ulp apart
rather than zero still receive distinct ranks and can still swap.

It was worth doing anyway because it is cheap, it is the statistically standard treatment,
and it makes the shaping a function of the scores rather than of an implementation detail of
the sort. It is not a fix for the conditioning problem, and presenting it as one would be the
same mistake as presenting the determinism flag as one.

**What it cost, measured after the fact.** Nothing in wall clock: regenerating Phase 0's 228
rank-shaped results ran at a median 1.01x of the originals, which is what an `O(n log n)`
sort either way predicts. It changed every one of those results, because ties are far denser
in Phase 0 than in Phase 2, and the changes were negligible: median absolute change in cosine
2.35e-07 against a typical cosine of 1.02e-02, with relative MSE improving in 218 of 228.
G0's verdict is unaffected by construction, since every comparison `gate.py` makes uses
`shaping=none`, which does no ranking.

One implementation note that only appeared under test: the first version was 1-D only, and an
`(n, episodes)` fitness then died with a `TypeError` from inside `concatenate` instead of the
`ValueError` from `tell` that names the fix. `tests/test_control.py` caught it. The shaping
contract is `(n,)` in and `(n,)` out, but the wrong shape has to be handed *through* rather
than rejected early, because `tell` owns that error message.

**If the conditioning problem itself needs fixing**, the two candidates are continuous
shaping (measured at 25x amplification, but a different optimizer) and quantising scores to a
multiple of `k` ulp before ranking, which converts near-ties into exact ties and, combined
with midranks, would make the update insensitive to perturbations below that scale. The
second is not standard practice and would need its own bias analysis before it went anywhere
near a result.
