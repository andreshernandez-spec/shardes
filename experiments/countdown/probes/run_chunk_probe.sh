#!/bin/bash
# Split the E13 rank-1 advantage into the perturbation scheme and the evaluation
# setting. `eval_chunk` reaches mirrored_seed only: run_es.py builds the low-rank
# arms as Mirrored(LowRank(r)) and never reads the config, so they always evaluate
# the whole population in one vmap. This times the seed arm at chunk 1, 5 and 15
# (15 = one vmap over every member on the device, the same batching LowRank does)
# against rank 1, on one host and one SHA.
#
# chunk must divide the per-side population, which Mirrored puts at 15.
# Diagnosis only; nothing here is cited as a result.
set -uo pipefail
cd "$(dirname "$0")/.."
. ../../.venv/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
for c in probe-seed-chunk1 probe-seed probe-seed-chunk15 probe-lr1; do
  echo "==== $c ($(date -u +%FT%TZ)) ===="
  python run_es.py --config probes/$c.yaml --seed 0
done
echo "PROBE_DONE"
