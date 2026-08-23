# Same-host timing probes

Thirty generations of one E13 arm, evaluation off past generation 0, so two arms can be
compared on one host, one commit and one jax build. They exist because the E13 clean
rerun put rank 1 24% above the 08-17 campaign and a cross-campaign difference cannot
tell a code change from a host: `probe-seed` and `probe-lr1` share everything but the
strategy, and `probe-lr1-padded` differs from `probe-lr1` only in the r=1 pad, which
`run_probes.sh` forces on with a local edit it reverts afterwards.

Diagnosis only. One seed, 28 steady-state updates, no seed statistics: nothing in the
paper cites these numbers, they explain a difference between numbers that are cited.

| set | commit | what the default build does on a GPU |
|---|---|---|
| `results-a100-pad-everywhere/` | 6d56af5 | pads r=1 (a671dc6) |
| `results-a100-pad-tpu-only/` | 030732d | no pad off TPU |

Median seconds per steady-state update, A100-SXM4-80GB, jax 0.11.1:

| probe | pad everywhere | pad TPU-only |
|---|---|---|
| full rank | 4.32 | 4.37 |
| rank 1, as the platform runs it | 2.89 | 2.56 |
| rank 1, the other build, forced | 2.49 | 2.95 |

Two readings. The pad costs the A100 15-16% per update, measured twice on two hosts
under both build directions, which is what made it TPU-only. And the host itself moves
full rank by about 1%, so the 24% that started this was the pad, not the machine.

Run on a pod whose campaign has finished:

    bash experiments/countdown/probes/run_probes.sh     # writes results-probe-*/
