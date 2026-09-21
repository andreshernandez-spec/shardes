# tools

Scripts that check the library, as opposed to experiments that produce a result. They
lived under `experiments/` until the two were told apart (docs/14), and they stay with
the library when the repository splits.

| script | what it answers |
|---|---|
| `mutation.py` | Does the suite notice when the library is broken on purpose? `python tools/mutation.py`, or `-k sobol` for a subset. Every gap it has found was a duplicated fact (docs/conventions.md). |
| `lowrank_compile_diag.py` | Where `tell`'s first-call cost goes per rank: trace time, HLO size, XLA compile time, run time. It is how the rank-16 compile went from 975 s to 71 s. |
| `bf16_probe.py` | The evidence behind the fitness dtype guard (docs/proposal-bf16-policy.md): 256 losses collapse to 2 distinct values in bfloat16. CPU, seconds. |
| `probe_lr1.py` | Why rank 1 is padded to a rank-2 dot on a TPU. Compiles for a v5e topology from any Linux machine, no TPU needed. |

None of them is imported by the package or shipped in the wheel.
