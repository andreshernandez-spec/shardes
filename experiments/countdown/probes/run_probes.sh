#!/bin/bash
# On a pod whose E13 campaign has finished: time the seed and rank-1 arms on the
# same host and SHA, then rank 1 with the r=1 pad forced on (the TPU program,
# a671dc6) by a local edit of the PAD_RANK1 switch, reverted afterwards. The
# 2026-08-23 run predates the switch and did the reverse (pad off by hand).
# Diagnosis only; nothing here is cited.
set -uo pipefail
cd "$(dirname "$0")/.."
. ../../.venv/bin/activate
git log --oneline -1
for c in probe-seed probe-lr1; do
  echo "==== $c ($(date -u +%FT%TZ)) ===="
  python run_es.py --config probes/$c.yaml --seed 0
done
echo "==== probe-lr1-padded ($(date -u +%FT%TZ)) ===="
sed -i 's/^PAD_RANK1: bool | str = "auto"/PAD_RANK1: bool | str = True  # PROBE/' ../../src/shardes/strategies/lowrank.py
git diff --stat
python run_es.py --config probes/probe-lr1-padded.yaml --seed 0
git checkout -- ../../src
git status --short
echo "PROBE_DONE"
