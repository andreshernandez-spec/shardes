"""Strategy A and Strategy B: the two ways to close the ES update loop across devices.

A. Scalar all-reduce, replicated regeneration.
   All-reduce the N fitness scalars (and the N seeds if not derivable, a few KB). Every
   device regenerates all N perturbations from seeds and contracts locally.
   Communication O(N). Contraction replicated D times.

B. Model-size all-reduce of the partial update.
   Each device contracts its local shard into a params-shaped partial, then psum.
   Communication O(d), same as data-parallel SGD. Contraction split D ways.

Both get implemented. The crossover in (N, d, D) is what Phase 2 measures and it is the
strongest single result in the paper (docs/05-paper.md C1).

Nothing public claims "ES only all-reduces scalars" until that measurement exists. The
claim is true for A and false for B, and both are legitimate.

Accumulate over members in f32 even when the perturbations are bf16. Summing 2^18 bf16
terms loses several digits. There is a test for this.
"""
