#!/bin/bash
# On a pod whose E13 campaign has finished: time the seed and rank-1 arms on the
# same host and SHA, then rank 1 with the r=1 pad (a671dc6) switched off by a
# local edit that is reverted afterwards. Diagnosis only; nothing here is cited.
set -uo pipefail
cd "$(dirname "$0")/.."
. ../../.venv/bin/activate
git log --oneline -1
for c in probe-seed probe-lr1; do
  echo "==== $c ($(date -u +%FT%TZ)) ===="
  python run_es.py --config probes/$c.yaml --seed 0
done
echo "==== probe-lr1-unpadded ($(date -u +%FT%TZ)) ===="
sed -i 's/if a.shape\[-1\] == 1:/if False:  # PROBE: pad off/' ../../src/shardes/strategies/lowrank.py
git diff --stat
python run_es.py --config probes/probe-lr1-unpadded.yaml --seed 0
git checkout -- ../../src
git status --short
echo "PROBE_DONE"
