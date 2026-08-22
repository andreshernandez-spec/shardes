# E16 stage-1 gate: the low-rank correction is not a prompt-batch accident

Two cells of `e16-gate.yaml` (frozen expectations in its header, committed at
71b3b28 before the run), one community A100-SXM4-80GB, ~2.5 h (~$3.50), pod
deleted after harvest. Everything identical to E15 except the frozen 8-prompt
batch: `puzzle_seed` 41 instead of 7.

## Verdict: PASS

| arm (N=240) | new batch | E15 (old batch) | raw prediction | ratio |
|---|---|---|---|---|
| full rank | 4.2e-4 [3.8-4.4] | 4.2e-4 [3.8-4.4] | 4.5e-4 | 0.93x |
| rank 1 | 3.9e-4 [3.8-4.2] | 3.9e-4 [3.8-4.2] | 7.6e-4 | 0.51x |

Both frozen expectations hold, and more tightly than required: medians and
ranges are indistinguishable from E15's on a different frozen batch. Full rank
sits at 0.93x of the raw fit (E15: 0.94x); rank 1 at 0.51x (E15: 0.52x,
frozen expectation ~0.5x). The ~0.53x low-rank correction is a property of
the model and estimator, not of the eight prompts it happened to be measured
on.

Per the gate's committed decision rule, stage 2 proceeds: Qwen2.5-1.5B on the
ORIGINAL E15 batch, full rank and mirrored rank 1, N in {30, 240}, five
seeds, with two predictions frozen before the run: the unmodified F5 fit, and
the fit times the 0.53x geometric-mean E15 correction for the low-rank arm.
