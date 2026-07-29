"""Coupled(inner, kind): sample design across members. kind in
{"none", "orthogonal_hd", "sobol_scrambled"}.

Phase 0 evaluates this unsharded, on one GPU, to answer Gate G0
(docs/01-phase0-estimator-harness.md). Making coupling survive sharding is Phase 3 and is
conditional on G0 (docs/04-phase3-coupling.md). Do not build the sharded path here until
G0 says yes.

orthogonal_hd: HD1 HD2 D3 ..., products of Hadamard transforms and Rademacher diagonals.
Approximately orthogonal in O(d log d) time and O(d) memory, because you cannot QR a
1e9 x 1e9 Gaussian. Built on transforms/fwht. Block-orthogonal per device with no
communication, which is why it is the good case under sharding.

sobol_scrambled: low-rank paths only. Full-rank sampling is in R^(mn) and every published
direction-number table stops around 20k dimensions, so the scheme is not constructible
there. See docs/01 C0.5.

Scrambling is mandatory, not optional: deterministic Sobol is biased at fixed N and the
estimate feeds SGD. A random digital shift is enough, XOR one uniform integer per
dimension. Derive the shift from base_key so it is shared across devices, and take the
point index from the global member index. Device-count invariance then falls out of the
construction.

Use the direct formulation, point i is the XOR of direction vectors selected by the set
bits of gray(i), not the sequential Gray-code recurrence. It is random-access by
construction, so skip-ahead across devices is free, and it is what vectorizes.

Direction numbers are the scarce part, not the algorithm. scipy.stats.qmc.Sobol carries
Joe-Kuo to 21,201 dimensions; dump the table once and embed it.

Whatever the kind, the point set must be identical for every device count.
"""
