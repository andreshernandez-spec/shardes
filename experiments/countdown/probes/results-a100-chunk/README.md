# How much of the E13 rank-1 advantage is the evaluation setting? A100, 2026-09-01

`run_chunk_probe.sh` at 1308696, one community A100-SXM4-80GB, jax 0.11.1, 30
generations per arm, one seed, all four arms back to back on the same host.
Diagnosis only, nothing here is cited as a result. The runs stamp
`dirty_worktree=true`; the only untracked paths are the probes' own output
directories and no tracked file differs, checked on the pod.

## The question

`eval_chunk` reaches `mirrored_seed` and nothing else. `run_es.py` builds it as
`Mirrored(SeedRegenerated(chunk=cfg["eval_chunk"]))`, and builds the low-rank arms as
`Mirrored(LowRank(r))`, which never reads the config. So E13's seed arm scored five
members per scan step while its rank-1 arm scored all fifteen per side in one vmap, and
the 41% between them contained that difference as well as the perturbation scheme.

`LowRank` cannot be made to scan: the base weight is unbatched under vmap so members
share one GEMM, which is the whole trick. The seed arm can be made to batch. So the
comparison is made fair from the seed side, by raising its chunk to 15, which is every
member on the device (Mirrored halves the population).

## Steady-state seconds per update, median over generations 2-29

| arm | s/update |
|---|---|
| seed, chunk 5 (as E13 ran it) | 4.313 |
| seed, chunk 15 (evaluation matched to the low-rank arm) | 3.619 |
| rank 1 | 2.520 |

Generations 0 and 1 are compilation (349-608 s) and excluded, as everywhere else.
The chunk-5 arm reproduces the E13 campaign's 4.45 s within host drift, and rank 1
its 2.61 s, so this host is measuring the same thing the campaign did.

## The split

| comparison | less time per update | speedup |
|---|---|---|
| as E13 ran it (chunk 5 against rank 1) | 42% | 1.71x |
| evaluation matched (chunk 15 against rank 1) | 30% | 1.44x |
| the chunk setting alone (chunk 5 against chunk 15) | 16% | 1.19x |

The two factors compose: 1.19 x 1.44 = 1.71.

**The headline survives, smaller.** Batching the seed arm's evaluation buys it 16%, so
about a quarter of the measured advantage was the evaluation setting rather than the
perturbation scheme. The remaining 1.44x is the perturbation: the low-rank arm still
does less work per update once both arms score their populations the same way.

Held-out reward is unmoved by the chunk, as it must be, since chunking changes only the
order of a sum: the two seed arms end 30 generations at 0.112 and 0.111 mean reward.

## Why E13 was not simply rerun this way

Chunk 5 is not a mistake in the E13 campaign, it is what fits alongside everything else
that campaign was doing, and 15 costs memory the seed arm's O(|params|) guarantee is
there to avoid. The point of this probe is not that E13 should have run at 15; it is
that the ratio it reports prices two paths as configured, and now the part of it that
is configuration is measured rather than estimated.

The same asymmetry, from the memory side rather than the time side, is
`../../results-e17b-memory/README.md`: batching members is what costs activations, which
is why the low-rank arms run out of HBM where the seed arm does not.
