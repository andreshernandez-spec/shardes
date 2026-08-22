#!/bin/bash
# E13 clean rerun: the same five ES arms and three seeds as
# results/e13-a100-2026-08-17, from a clean checkout at a committed SHA, so
# every log stamps dirty_worktree=false. Committed before the run, cited by
# SHA. Run on one A100 per invocation; split the arms across pods:
#
#   bash e13_campaign.sh seed                 # mirrored-seed, seeds 0-2
#   bash e13_campaign.sh lr1 lr1-frozen-embed # two arms, seeds 0-2 each
#   bash e13_campaign.sh lr4 lr16
#
# Each run writes results-es-<arm>-s<seed>/{log,eval}.jsonl via run_es.py;
# harvest.sh collects them under results/e13-a100-<date>-clean/ with the
# E13 file names so plot_e13.py and tb3.py read the new directory unchanged.
set -euo pipefail
cd "$(dirname "$0")"
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "worktree is dirty; the point of this rerun is a clean one" >&2; exit 1; }
git log --oneline -1
for arm in "$@"; do
  cfg=$([ "$arm" = seed ] && echo pilot.yaml || echo "pilot-$arm.yaml")
  for s in 0 1 2; do
    echo "==== $arm seed $s ($(date -u +%FT%TZ)) ===="
    python run_es.py --config "$cfg" --seed "$s"
  done
done
echo "E13_CAMPAIGN_DONE $*"
