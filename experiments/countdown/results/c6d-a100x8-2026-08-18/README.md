# C6d: determinism and device-count invariance at 0.5B, D in {1,8}. 2026-08-18

Five runs on one 8x A100-SXM4-80GB pod (secure cloud, `env.txt`), configs
`c6d-*.yaml` at commit 87195b6, population 32 (Mirrored's pairs cannot shard 30
over 8 devices; this is a methods demonstration, not a training result).
`compare_c6d.py` computes the raw comparisons and `c6d_decompose.py` is the
follow-up that separates what the raw numbers conflate; `decompose-verdicts.txt`
holds its output verbatim.

What C6d turns out to mean at bf16, stated as three separate claims:

1. **Same program, same seed: bitwise determinism, across processes.** Two
   independent 20-generation D=8 runs agree byte for byte in every logged reward
   and exactly (max relative error 0.0) in all 494M final parameters, and a third
   D=8 process reproduces generation 0's mean reward to the last bit
   (0.0277343765). No PPO/GRPO trainer offers this.
2. **The update path is device-count invariant at tolerance.** With one fixed
   fitness vector, the D=1 and D=8 updates on the full 0.5B tree agree to
   6.3e-06 norm relative error (tolerance 1e-5, invariant 2); the worst single
   leaf sits at 7.9e-04 on a per-leaf scale, the small-norm-leaf effect.
3. **End-to-end trajectory equality across different programs is not a bf16
   property, and the demo measured why.** The D=1 and D=8 20-generation runs
   diverge from generation 0 (mean reward 0.0250 vs 0.0277) even though the
   decomposition shows a D=1 and a D=8 program producing bit-identical tokens for
   all 32 members in one process: greedy argmax at bf16 flips near-ties under
   different XLA compilations (fusion and autotuning choices vary per program),
   and a flipped token is a different sampled trajectory. The divergence tracks
   the compiled program, not the device count; the same effect would follow an
   XLA upgrade. So the honest methods statement is per-program bitwise
   reproducibility plus per-update invariance at float tolerance, and the
   trajectory-level caveat is stated rather than hidden.

The gate run also happens to be the library's first execution on real multi-GPU
hardware: mesh, shard_map and NCCL collectives at 0.5B worked unmodified, first
try, after the config-level chunk divisibility fix the gate itself caught.
