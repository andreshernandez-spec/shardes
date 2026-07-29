"""Real-hardware check that the simulated-device shortcut did not lie.

Marked @pytest.mark.gpu, deselected by default, run by hand before each gate. A 1-GPU and a
2-GPU run reproducing test_device_invariance against the CPU-simulated result.

Do this before Phase 2, not during. Set
XLA_FLAGS=--xla_gpu_deterministic_ops=true for the comparison.
"""
