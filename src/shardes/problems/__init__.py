"""Test objectives with exact or backprop gradients.

The oracle is the point of Phase 0: use a differentiable model so backprop gives the true
grad f directly. No proxy metric, no reference-estimator-with-huge-N, no ambiguity about
what "good" means (docs/01-phase0-estimator-harness.md C0.4).

Three, in increasing realism: quadratic, mlp, transformer_block.

These are library code rather than experiment code because both tests/ and experiments/
import them.
"""
