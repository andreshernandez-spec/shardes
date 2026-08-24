#!/bin/bash
# E18 L1 (docs/10 section 4): the NCCL smoke on one 2-GPU pod. The
# single-process 1x2 run is the anchor (writes the invariance reference),
# then 2x1 runs the same preflight and driver as two processes with one
# GPU each through jax.distributed, which is what the cluster does across
# hosts. Run detached on the pod from this directory with the venv active:
#   setsid nohup bash l1.sh > /root/l1.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")"
export E18_RESULTS_DIR=results-e18-l1
phase() { echo; echo "==== $1 ($(date -u +%FT%TZ)) ===="; }

phase "preflight 1x2 (anchor)"
E18_TOPOLOGY=1x2 E18_EXPECT_DEVICES=2 timeout 900 python preflight.py

phase "preflight 2x1 (two processes, one GPU each)"
CUDA_VISIBLE_DEVICES=1 E18_TOPOLOGY=2x1 E18_COORD=localhost:12377 E18_NPROC=2 \
  E18_PID=1 E18_EXPECT_DEVICES=2 timeout 900 python preflight.py \
  > l1-preflight-1.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=0 E18_TOPOLOGY=2x1 E18_COORD=localhost:12377 E18_NPROC=2 \
  E18_PID=0 E18_EXPECT_DEVICES=2 timeout 900 python preflight.py
wait $p1

phase "driver 1x2"
E18_TOPOLOGY=1x2 timeout 1800 python driver.py --config e18-l1.yaml

phase "driver 2x1"
CUDA_VISIBLE_DEVICES=1 E18_TOPOLOGY=2x1 E18_COORD=localhost:12378 E18_NPROC=2 \
  E18_PID=1 timeout 1800 python driver.py --config e18-l1.yaml \
  > l1-driver-1.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=0 E18_TOPOLOGY=2x1 E18_COORD=localhost:12378 E18_NPROC=2 \
  E18_PID=0 timeout 1800 python driver.py --config e18-l1.yaml
wait $p1

phase "DONE"
echo "L1_DONE"
