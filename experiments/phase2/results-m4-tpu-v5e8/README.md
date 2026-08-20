# M4 on TPU v5e-8: E9's TPU column, 2026-08-20

One T5 session (`kaggle/t5m4/`, pinned e9aa7b7), `m4.py --config sweep-tpu.yaml` at
D=1 and D=8, the same four shapes and protocol as the A100 runs (`results-m4-a100-*`).
Both references were available: evosax from PyPI, EGGROLL (hyperscalees) from the
authors' repo at b77f7d6, installed --no-deps, run unmodified via m4.py's loader.
Every row stamps env and commit; failures are recorded as rows.

What TB1's TPU column says, from `m4-d1.json` / `m4-d8.json`:

- **Within EGGROLL's own throughput at D=1**: `shardes/mirrored_lr1/B` runs at 0.68x
  to 0.85x of `eggroll/rank1` across the four shapes (e.g. 23.3M vs 27.3M tok/s at
  d=512 N=256; 3.2M vs 4.5M at d=2048 N=256). docs/05's framing applies verbatim:
  within their throughput, with a general API, is the result.
- **D=8 is what the library exists for**: none of the references shards, so their
  D=8 rows equal their D=1 rows, while `mirrored_lr1/B` reaches 149.7M tok/s at
  d=512 N=1024, six times the reference's single-chip ceiling. `m4.py` marks the
  unsharded rows so nobody reads that as a like-for-like ratio.
- **The evosax foreclosure is now a measured row**: `Open_ES` OOMs at d=2048 on a
  16 GB chip at every device count (ravel_pytree materializes the (N, n_params)
  population), recorded as failures in the JSONs. On the A100 it merely lagged;
  on the v5e it does not fit.
- `seed_regenerated/B` is regeneration-bound at D=1 on this platform (consistent
  with the E8 surface) and scales about 8x to D=8.

Regenerate: push `kaggle/t5m4/` at a commit containing this directory's parent
config (fill the username at push time), `--accelerator TpuV5E8`.
