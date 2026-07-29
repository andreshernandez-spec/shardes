"""The PerturbationStrategy protocol: sample / apply / contract.

The interface everything else hangs off. Sketched in docs/01-phase0-estimator-harness.md
C0.1; refine it there, do not copy the sketch.

    sample(key, params, n) -> Perturbation
        Opaque per-member state for n members. Shape-aware: leaves keep their (m, n)
        structure.

    apply(params, pert, sigma) -> Callable
        A callable evaluating the model for all n members. Full rank materializes per
        member or regenerates from seed. Low rank rewrites x @ W.T into
        x @ W.T + (x @ B) @ A.T and never materializes.

    contract(pert, weights) -> PyTree
        Contract shaped fitness weights (n,) into a params-shaped update. The only place a
        full (m, n) tensor is instantiated.

A Protocol, not an ABC. Structural typing keeps user-defined strategies first-class
without inheritance (docs/conventions.md).

The perturbation scheme cannot be a parameter: it determines how the forward pass is
structured. That is why it owns all three steps (docs/00-context.md).
"""
