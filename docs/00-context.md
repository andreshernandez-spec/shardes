# 00 — Context

Everything here is background. No task in this file. If you're picking up work, read this
once, then go to `PLAN.md`.

---

## The two papers

Both landed in late 2025, both claim ES works at billion-parameter scale, and they are
**not the same algorithm**. Sarkar et al. cite Qiu et al. as motivation; the work is
concurrent and independent.

### Qiu et al. — *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*

arXiv [2509.24372](https://arxiv.org/abs/2509.24372) · Cognizant AI Lab ·
code: `VsonicV/es-fine-tuning-paper`, framework: `VsonicV/es-at-scale` · PyTorch + Ray + vLLM

No new algorithm. Simplified NES / OpenAI-ES, unchanged, with **full-rank unstructured
Gaussian perturbations**. What makes it fit in memory is engineering: store only the random
**seeds**, regenerate the noise on the fly for both perturbation and update, and apply
perturb/restore **layer-by-layer in place**.

- Population **N = 30** (matched to GRPO's), versus the 10,000+ of classical ES on far
  smaller models.
- Claims: first full-parameter ES fine-tuning at billion scale without dimensionality
  reduction; beats PPO/GRPO on Countdown across Qwen2.5 and LLaMA 0.5B–8B; under 20% of
  RL's training samples; lower run-to-run variance; no observed reward hacking; better
  reward/KL Pareto front without an explicit KL penalty.

### Sarkar et al. — *Evolution Strategies at the Hyperscale* (EGGROLL)

arXiv [2511.16652](https://arxiv.org/abs/2511.16652) · Oxford + MILA + **NVIDIA** ·
v1 Nov 2025, v2 Feb 2026 · JAX

New algorithm. The bottleneck attacked is **arithmetic intensity**, not memory footprint.
Each perturbation is `E = (1/√r)·ABᵀ` with `A ∈ ℝ^{m×r}`, `B ∈ ℝ^{n×r}`, `r ≪ min(m,n)`,
and is **never materialized**:

```
base = x @ W.T
out  = base + (x @ B) @ A.T        # one shared GEMM + two thin GEMMs
```

All population members share the base activations, so the population runs as *batched
inference*. Storage drops `mn → r(m+n)` per layer; forward cost likewise.

- The update is an average over `N` members, so it stays high-rank even though each
  perturbation is rank-`r`. Convergence to the full-rank ES update at `O(1/r)`.
- **r = 1 is enough in practice.**
- Population **64 → 262,144** in the integer-pretraining sweep; up to 91% of pure batch
  inference throughput; ~100× naive ES at billion scale.
- Also: stable pretraining of a pure-integer recurrent LM; competitive with GRPO on
  Countdown and GSM8K with RWKV-7.

### The tradeoff, stated cleanly

|  | Qiu et al. | Sarkar et al. (EGGROLL) |
|---|---|---|
| Perturbation | full-rank, unstructured | rank-`r` factored, never formed |
| Memory trick | regenerate from seed, in place | don't form it at all |
| Population | 30 | 64 → 262,144 |
| Members share a GEMM? | **no** — each is a different weight set | **yes** |
| Headline | sample efficiency vs PPO/GRPO | 91% of batch-inference throughput |

These are opposite bets on one tradeoff. Qiu's seed trick preserves full-rank noise but
forces members to be evaluated as separate forward passes, because each member genuinely
*is* a different weight matrix. EGGROLL gives up per-member full-rank noise precisely to
buy the shared base GEMM. You can't have both.

**Design consequence for this library**: the perturbation scheme cannot be a parameter. It
determines how the forward pass is structured. It has to be a strategy that owns
sample / apply / contract.

**One synthesis worth noting**: the two tricks are alternatives at the *sample* step but
composable at the *contract* step. For large `N`, storing `A` and `B` for all members costs
`N·r·(m+n)` — at `m=n=4096, N=2^18, r=1` that's ~2 GB/layer in bf16. Regenerating `A`, `B`
from per-member seeds during contraction (Qiu's trick applied to EGGROLL's structure)
removes that. Nobody has written this down; it falls out naturally from having both
strategies in one codebase.

---

## The gap in the ecosystem

`evosax` is the de-facto standard JAX ES library, 30+ algorithms. Verified problems:

1. **Everything is flattened to one dense vector.** `evosax/algorithms/base.py`:
   `self.num_dims = self.solution_flat.size` via `ravel_pytree`. This is the architectural
   blocker — it makes per-matrix structured perturbation (i.e. EGGROLL), parameter
   sharding, and pytree-native ES all impossible without touching the base class.
2. **Zero sharding code.** Grepping for `shard_map|jax.sharding|NamedSharding|pmap|Mesh(|PartitionSpec`
   returns exactly one hit, an unrelated `device_put`. Multi-device support is
   documentation only: a notebook where the user passes `out_shardings` by hand. That
   shards the population but **replicates all algorithm state** — for CMA-ES, the `d×d`
   covariance sits on every device.
3. **Stale JAX pin**: `jax>=0.5.0,<0.7` while JAX is at 0.11.
4. **Deprecated Brax path**: targets `brax.envs`, deprecated at Brax v0.13.0 in favour of
   MuJoCo Playground.

Elsewhere: `google/evojax` is archived (read-only since 2025-08-29). `EMI-Group/EvoX`
migrated to PyTorch. `EMI-Group/evorl` has the best sharding in the space (a real
`evorl/distributed/` package on modern `shard_map`) but is scoped to PBT and CEMRL
workflows, not a general ES `ask`/`tell`. EGGROLL's own multi-GPU scripts import
`jax.experimental.shard_map`, deprecated in JAX 0.8.0.

Items 3 and 4 are independently mergeable warm-up PRs to evosax — it merges outside PRs
routinely. Worth doing in parallel; they are not part of this project's core claim.

---

## Prior art on coupling / QMC for ES

Relevant to Phase 0 only, now: Phase 3 was dropped when G0 came back negative (`docs/04`). Kept because it is the reasoning the negative result is read against. Summary of the search: the idea is old, works in low
dimensions, and the naive scale-up fails for a reason usually stated incorrectly.

**Classical, low-dimensional, works.** A twenty-year line: Auger, Jebalia & Teytaud (2005)
replaced Gaussian mutations with low-dispersion quasi-random points and proved linear
convergence rates for global rather than merely local optimization; Teytaud & Gelly's DCMA
(GECCO 2007) swapped CMA-ES's independent Gaussian mutations for a quasi-random sample;
Teytaud (2015) showed quasi-random numbers improve CMA-ES on the BBOB testbed; Teytaud &
Teytaud (GECCO 2016) extended it across SA, CMSA and CMA with clear gains, strongest on
multimodal problems. All at `d ≤` a few hundred.

**The naive scale-up fails, but not because of Koksma–Hlawka.** The `(log N)^d / N` bound
is famously pessimistic and low effective dimension rescues QMC at nominal dimension 360 in
finance routinely. The operative obstacle is the ratio `N/d`:

Write the antithetic ES estimator as `ĝ = M∇f` with `M = (1/N) Σ εₙεₙᵀ`. For `N ≪ d`, `M`
has rank `N`, so the dominant error is the component of `∇f` outside `span{εₙ}` — and that
span is a uniformly random `N`-dimensional subspace whether you sample i.i.d. or with
perfect coupling. Coupling only flattens the spectrum *within* the span, and i.i.d.
Gaussians in `d` dimensions already have pairwise cosines of order `1/√d`. **The relative
gain from enforcing orthogonality scales like `N/d`.** At `d = 10⁹`, `N = 10⁶`: `10⁻³`.
Provably positive, practically nothing.

**At full rank, what survives is coupling, not low-discrepancy sequences.** Choromanski, Rowland,
Sindhwani, Turner & Weller, *Structured Evolution with Compact Architectures for Scalable
Policy Optimization* (ICML 2018, arXiv 1804.02395) — blackbox gradient approximation with
structured random orthogonal matrices, provable MSE improvement over i.i.d. Rowland et al.,
*Geometrically Coupled Monte Carlo Sampling* (NeurIPS 2018) is the general framing. Their
appendix covers Hadamard–Rademacher directions **and** QMC strategies. The orthogonal route
won at scale because you can't QR a `10⁹ × 10⁹` Gaussian: they use `HD₁HD₂…` products of
Hadamard transforms with Rademacher diagonals, approximately orthogonal in `O(d log d)`
time and `O(d)` memory.

That conclusion is scoped to the full-rank setting it was drawn in, and it is why the
full-rank arm of this project carries `orthogonal_hd` and nothing else. Under low rank the
sampling dimension collapses to `m + n` and digital nets come back into range. See the open
angle below.

**Note for the wider project**: that construction *is* a fast Walsh–Hadamard transform. If
the FWHT kernel is on the roadmap for other reasons (quantization rotations, SRHT), this is
a third independent consumer of the same primitive.

**The open angle.** EGGROLL inverts the `N/d` arithmetic. Under rank-1 perturbation you
don't sample in `ℝ^{mn}`; you sample `a ∈ ℝ^m` and `b ∈ ℝ^n`, so the per-layer sampling
dimension is `m + n ≈ 8k` for a 4096-wide layer, not 16M. At `N = 2^18`, `N/d_eff ≈ 32`
instead of `10⁻³`. As far as the searches went, this is the first LLM-scale ES whose
population exceeds its sampling dimension — the regime where sample design has leverage.
Hypothesis: coupling `{aₙ}` across `S^{m−1}` and `{bₙ}` across `S^{n−1}` tightens the
constant in EGGROLL's `O(1/r)` rate. Untested — EGGROLL is eight months old, and the
adjacent ZO variance-reduction literature (P-GAP, LOREN, GRZO) all attacks dimension via
subspace projection or preconditioning, never via sample design.

**Three obstacles to hold onto.**

1. Mirrored/antithetic sampling is already standard in ES and already cancels odd-order
   terms. The baseline is not i.i.d.; much of the easy win is already spent.
2. Rank-based fitness shaping — the thing that makes ES work at all — is **discontinuous in
   ε**. QMC's advantage rests on bounded Hardy–Krause variation, which rank transforms
   destroy. Measure with and without shaping; expect the shaped case to degrade toward MC
   rates.
3. Sobol is a low-rank-only scheme here, and that's a constraint, not a preference.
   Full-rank sampling is in `ℝ^{mn}`, and every published direction-number table stops
   around 20k dimensions (cuRAND documents 20,000; `scipy.stats.qmc.Sobol.MAXDIM` is
   21,201). Extending Joe-Kuo past that is its own research problem. Under rank-1 the
   sampling dimension is `m + n`, about 1k for the Phase 0 block and 8k for a 4096-wide
   layer, comfortably inside the tables and within an order of magnitude of the
   `d ≤ few hundred` regime where the classical results were obtained.

   The dimension-grading complaint (early coordinates get better equidistribution, and
   unlike finance there's no Brownian-bridge analogue to order NN parameters by importance)
   is real but much weaker at that scale, and it changes character. In finance the problem
   is that importance structure exists and you have to find the ordering. Here the
   coordinates of `a ∈ ℝᵐ` index a layer's output units, which are close to exchangeable,
   so there's no ordering being left on the table. Grading costs a little uniformly instead
   of failing systematically.

   Scrambling is still mandatory. Deterministic Sobol is biased at fixed `N`, the estimator
   feeds SGD, and unbiasedness isn't optional. A random digital shift is enough.

**The caveat that matters most.** Lower estimator variance does not straightforwardly mean
better ES. The Gaussian-smoothing story — noise in parameter space smooths a jagged reward
landscape — implies the noise is doing optimization work, not just adding error. A
better-conditioned estimate could be a worse smoother, and coupling narrows exploration in
a way that may hurt on multimodal objectives. This is exactly why the older QMC-for-ES
results are strongest on multimodal problems, and why an MSE plot is **not** a proxy for
task performance. Phase 0 gates the *abstraction*; only end-to-end runs can settle the
algorithmic claim.

---

## Reading list, in order

1. Sarkar et al., arXiv 2511.16652 — §4 (algorithm), §5 (theory), Appendix F (throughput).
2. Qiu et al., arXiv 2509.24372 — Algorithms 1 and 2.
3. Choromanski et al., arXiv 1804.02395 — §3 and the appendix on HD constructions.
4. `evosax/algorithms/base.py` — read `num_dims` and understand why it forecloses (1).
