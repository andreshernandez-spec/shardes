"""LowRank(r): EGGROLL. Perturbation E = (1/sqrt(r)) A B^T, never materialized.

    A: (n_members, m, r)
    B: (n_members, n, r)

apply rewrites the layer so all members share one base GEMM:

    base = x @ W.T
    out  = base + (x @ B) @ A.T

contract is sum_n w_n a_n b_n^T = (A * w) B^T, one (m x Nr) by (Nr x n) GEMM, m*n*N*r
FLOPs. At m = n = 4096, N = 2^18, r = 1 that is 4.4 TFLOP, about 6 ms on an H100. The cost
that bites is storage: A and B together are N*r*(m+n), roughly 2 GB per layer in bf16.
That threshold is what motivates regenerating A and B from seeds during contraction, the
synthesis noted in docs/00-context.md.

Invariant 3 (CLAUDE.md): if a profile shows an (n_members, m, n) array under this path,
the implementation is wrong. There is a jaxpr test for it.

Reductions over members accumulate in f32 even when A and B are bf16.
"""
