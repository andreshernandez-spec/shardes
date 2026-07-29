"""One transformer block. The configuration the Phase 0 result is actually about.

Pick shapes so N/d_eff sweeps three orders of magnitude on one GPU. Suggested m = n = 512,
so d_eff = m + n = 1024 and N in {2^6 ... 2^18} gives N/d_eff in [0.06, 256]. At N = 2^18
that is 262144 * 512 * 4 B, about 0.5 GB each for A and B in f32. Comfortable.

Do not jump straight to m = n = 4096. You will be memory-bound before you are in the
interesting regime.

A single transformer block is not an LLM. The N/d_eff regime transfers, the loss landscape
does not, and that limitation gets stated up front.
"""
