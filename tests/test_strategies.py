"""Per-strategy structural correctness.

    test_contract_matches_naive     vectorized contract == explicit Python loop
                                    sum_n w_n a_n b_n^T, at small N
    test_lowrank_matches_reference  against a naive materialize-everything implementation,
                                    small m, n, N
    test_pytree_structure_preserved update tree structure == params tree structure, leaf
                                    shapes match

Tolerances are stated, not discovered: f32 against an exact oracle is rtol=1e-6, bf16
paths are rtol=1e-2 and say so.
"""
