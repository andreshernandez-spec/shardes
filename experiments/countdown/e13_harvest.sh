#!/bin/bash
# Collect run_es.py output directories into one E13-shaped results directory:
#   bash e13_harvest.sh results/e13-a100-<date>-clean [results-es-*]
# Each results-es-<arm>-s<seed>/ becomes es-<arm>-s<seed>-{log,eval}.jsonl in
# the destination, the names plot_e13.py, tb3.py and summarize.py read.
set -euo pipefail
cd "$(dirname "$0")"
dest=${1:?destination directory}; shift
mkdir -p "$dest"
for d in "${@:-results-es-*}"; do
  [ -d "$d" ] || continue
  name=${d#results-}            # es-<arm>-s<seed>
  cp "$d/log.jsonl" "$dest/$name-log.jsonl"
  cp "$d/eval.jsonl" "$dest/$name-eval.jsonl"
  echo "$d -> $dest/$name-{log,eval}.jsonl"
done
ls "$dest" | wc -l
