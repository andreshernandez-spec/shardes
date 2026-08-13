# Fixes for the 2026-08-11 review: four suggestions

Drafts, not decisions. Ground rule 1 puts the strategies, the sharding logic and the
estimator math in Andres's hand; this is the reasoning and the options, with the arithmetic
checked where it is checkable.

Every claim below was reproduced against JAX 0.11.0 on CPU before being written down.

---

## 1. `LowRank` and per-coordinate sigma

**This is not a bug to patch. It is a contract that promises something the representation
cannot express.**

`core.State` documents sigma as "either a scalar or a params-shaped pytree". For a full-rank
strategy that is fine. For `LowRank` the perturbed weight is `W + sigma * (A B^T)` with `*`
elementwise, and the whole design rests on never forming that sum: `apply_to` computes
`x @ W.T + ((x @ B) @ A.T) * scale`, two GEMMs against the factors.

A general per-coordinate sigma breaks the identity because **it does not preserve the rank**.
Measured, `m=5, k=4, r=2`: `rank(sigma * (A B^T)) = 4`, not 2. There is no two-GEMM form to
rescue, because the object being described is no longer low rank.

So the honest options are:

**(a) Reject it, and say why.** At `ShardedES.__init__`, if any leaf of a tree-valued sigma
corresponds to a leaf the strategy will structure, raise. Cheap, immediate, and turns a
broadcasting error from deep inside `apply_to` into a sentence at construction. The error
should name the reason rather than the shape, because the shape is not the problem:

    per-coordinate sigma is not expressible under LowRank. The perturbation is
    W + sigma * (A B^T), and an elementwise sigma raises the rank of that product above r,
    so it cannot be applied as two GEMMs against the factors. Use a scalar sigma, or a
    separable one (see below), or a full-rank strategy.

**(b) Support the separable family, which is exactly the part that does work.** If sigma
factorises as an outer product `u v^T` then

    (u v^T) * (A B^T) == (u[:, None] * A) @ (v[:, None] * B).T

which is still rank `r` and still two GEMMs. Verified numerically. That family is not a
curiosity: per-output-unit and per-input-unit step sizes are the per-coordinate schedules
anyone actually asks for, and both are separable. It would mean a sigma leaf for a
structured weight is a pair `(u, v)` rather than an `(m, k)` matrix, which is a type change
in the public contract and wants thinking about rather than bolting on.

**(c) Materialize when sigma is non-scalar.** Correct, and it silently discards invariant 3.
Mentioned only to be rejected.

Recommendation: **(a) now, (b) as a considered follow-up.** (a) is a guard and a docstring
correction; (b) is a feature with an API consequence.

`gather` needs the same treatment either way. `(W + sigma * A B^T)[ids]` is
`W[ids] + sigma[ids] * (A[ids] B^T)`, and the current code does not index the scale. Under a
scalar it does not matter, which is why nothing failed; under (b) it would.

---

## 2. `Mirrored` and the seed contract

    base_ids = member_ids[0::2] // 2

The slice is positional and the comment above it says the opposite: "the direction index
comes from the id rather than from the position in the batch". Measured, the same global
member changes meaning with batching:

| ids passed | direction | member 1 is |
|---|---|---|
| `[0, 1]` | 0 | the **negative** image |
| `[1, 2]` | 0 | the **positive** image |

`sharding.py` states the contract this violates: "member i's perturbation derives from
`fold_in(base_key, i)`, where i is the global member index. Never the device index, never a
per-device counter, never sequential consumption." Position in the batch is a fourth thing
the contract did not think to forbid.

Safe through `ShardedES` today, because `member_ids` is contiguous and `check_population(
paired=True)` forces an even count per device, so every shard starts on an even id. Unsafe
for direct strategy use, and unsafe for any future chunking that does not preserve alignment.

**The obvious fix, validation, does not work under `jit`.** `member_ids` is a traced array,
so `if (ids[0::2] % 2 != 0).any(): raise` is not expressible. That rules out the
straightforward guard and is worth knowing before reaching for it.

**(a) Derive both direction and sign from each id.** Correct for any batch:

    directions = member_ids // 2          # per member, not per pair
    signs = jnp.where(member_ids % 2 == 0, 1.0, -1.0)

and sample the inner perturbation on `directions`. The cost is real: the inner `sample` now
sees `n` ids rather than `n/2`, so a materializing inner strategy builds `n` perturbations
where it used to build `n/2`. Paired members would hold two identical copies. For
`SeedRegenerated` this is free (it regenerates anyway); for `IIDGaussian` and `LowRank` it
doubles the perturbation memory, which is the resource the whole library is about.

**(b) Keep the `n/2` fast path and make alignment a precondition enforced where it is
static.** `ShardedES` knows `n` and the mesh at construction, so `check_population` can
assert what `Mirrored` needs, and `Mirrored.sample` can assert it host-side when it is handed
concrete ids and stay silent when traced. The docstring then states the precondition instead
of asserting the opposite.

**(c) Make pairing a strategy capability rather than an `isinstance` check.** The review
raised this separately and it is the same decision: `ShardedES.__init__` hardcodes
`isinstance(strategy, Mirrored)`, so a user-defined paired strategy cannot ask for the
alignment it needs. A `pairing` attribute the protocol reads would fix the review's
extensibility complaint and give (b) somewhere honest to live.

Recommendation: **(b) plus (c)**, and fix the comment first regardless. (a) is the only
option that is correct by construction, but it pays memory in the common case to protect
against a batching pattern the library never produces, and this project has consistently
chosen the explicit precondition over the defensive cost.

---

## 3. `check_population` and sigma validation

Smallest and least interesting, and the one most likely to bite a user first.

- `n = 0` is accepted, and `tell` then divides by `n * sigma`, producing all-NaN parameters.
  Add `if n < 1: raise`. A population of zero is never a configuration anyone meant.
- `sigma = 0` divides the same way. Validate at `init`: a scalar sigma must be finite and
  strictly positive; a tree-valued sigma must match `params`' structure, leaf for leaf.
- `strategies._scale.per_leaf` treats a *mismatched* tree as a scalar and broadcasts the
  whole object. That is the silent half: a user whose sigma tree has one wrong key gets a
  plausible run rather than an error. Structure validation at `init` closes it, and
  `per_leaf` should stop guessing.

None of this affects the committed benchmarks, which use a scalar sigma throughout.

---

## 4. `shaping`: non-finite fitness and rank dtype

Two separate defects in one function.

**Non-finite fitness.** `_midranks` compares with `!=`, and `NaN != NaN`, so every failed
member becomes its own tie group. Measured, `centered_ranks([0, nan, nan, 1])` gives
`[-0.5, 0.167, 0.5, -0.167]`: the two NaNs get **different** weights, and one of them gets
`+0.5`, the largest in the population.

The direction matters more than the tie. A member whose episode diverged is not a good
member, and under a rank shaping it is currently receiving the strongest vote available. Any
RL or control workload can produce this, and `tests/test_control.py` drives MuJoCo.

The property to fix to is: **non-finite fitness must be treated as the worst outcome, and
all non-finite members must tie.** Mapping `NaN` to `+inf` before ranking gets both in one
line, and follows the existing sign convention rather than restating it:

    f = jnp.where(jnp.isnan(f), jnp.inf, f)

`+inf` because `tell` descends on the objective, so `+inf` is the worst loss. Worth an
explicit test in both directions, since a sign error here is invisible: the run still trains,
just away from the members that worked.

Whether to *exclude* non-finite members instead (weight exactly zero) is a real alternative.
It is more conservative and it changes the estimator's normalisation, which is why it is a
decision rather than a patch.

**Rank dtype.** Positions are built at `fitness.dtype`, so a bf16 fitness produces bf16 rank
weights. bf16 has 8 mantissa bits: above ~256 members the rank positions themselves stop
being distinct, and the shaping quietly compresses. Compute positions and weights in float32
regardless of input dtype and cast at the end if anything needs it. This is unrelated to the
NaN issue and cheaper to fix.

---

## Not covered here

The dtype policy (bf16 params becoming float32 after one update) is a broader decision than
these four: master-weights-in-f32 and preserve-input-dtype are both defensible, they differ
in memory and in whether a multi-generation `lax.scan` can carry the state at all, and the
perturbation dtype should probably become an explicit choice rather than a consequence. It
deserves its own note.

`H2`, pytree-aware objectives silently computing the wrong function under `LowRank`, is not
a fix suggestion because it is not a fix. It is a question about what the library promises:
whether a user objective may treat `params` as an ordinary pytree. Answering "no" needs a
guard and a documented restriction; answering "yes" needs `LowRankWeight` to stop being a
transparent pytree to user code, which is a different library. `tests/test_core.py::sphere`
takes the first answer implicitly today, by using `jax.tree.leaves` and passing.
