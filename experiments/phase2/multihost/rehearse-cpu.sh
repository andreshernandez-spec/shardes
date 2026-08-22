#!/bin/bash
# E18 L0: the FULL stack, two processes x 4 simulated CPU devices, free.
# Order matters and mirrors the cluster session: 1x8 preflight (writes the
# invariance reference), 2x4 preflight (compares), then the driver in both
# topologies on a tiny config. Any non-zero exit is a failed rehearsal.
set -e
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu
COORD=localhost:12390

echo "== preflight 1x8 (single process, 8 simulated devices) =="
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  E18_TOPOLOGY=1x8 E18_EXPECT_DEVICES=8 E18_RESULTS_DIR=results-e18-rehearsal \
  python preflight.py

echo "== preflight 2x4 (PID 1 from an ISOLATED copy: node 1 has no shared disk) =="
# Node 1 gets what launch.sh gives it on the real cluster: a FULL fresh
# clone of the repo at this commit, and nothing else. A bare file copy
# once passed preflight here while the driver, which imports harness from
# the tree, would have died on the real node 1.
REPO_ROOT=$(git rev-parse --show-toplevel)
NODE1_ROOT=$(mktemp -d)
git clone -q "file://$REPO_ROOT" "$NODE1_ROOT/shardes"
( cd "$NODE1_ROOT/shardes" && git checkout -q $(git -C "$REPO_ROOT" rev-parse HEAD) ) 2>/dev/null || true
NODE1_SIM="$NODE1_ROOT/shardes/experiments/phase2/multihost"
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=$COORD E18_NPROC=2 E18_PID=0 \
  E18_EXPECT_DEVICES=8 E18_RESULTS_DIR=results-e18-rehearsal \
  python preflight.py > /tmp/e18-preflight-0.log 2>&1 &
( cd "$NODE1_SIM" && \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=$COORD E18_NPROC=2 E18_PID=1 \
  E18_EXPECT_DEVICES=8 E18_RESULTS_DIR=results-e18-rehearsal \
  python preflight.py > /tmp/e18-preflight-1.log 2>&1 ) &
wait
grep "PREFLIGHT PASS" /tmp/e18-preflight-0.log

echo "== driver 1x8, tiny config =="
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  E18_TOPOLOGY=1x8 python driver.py --config e18-rehearsal.yaml \
  --skip-preflight-check

echo "== driver 2x4, tiny config (resume check: rerun skips) =="
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=localhost:12391 E18_NPROC=2 E18_PID=0 \
  python driver.py --config e18-rehearsal.yaml --skip-preflight-check \
  > /tmp/e18-driver-0.log 2>&1 &
( cd "$NODE1_SIM" && \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=localhost:12391 E18_NPROC=2 E18_PID=1 \
  python driver.py --config e18-rehearsal.yaml --skip-preflight-check \
  > /tmp/e18-driver-1.log 2>&1 ) &
wait
grep "topo=2x4" /tmp/e18-driver-0.log | tail -4
for pid in 0 1; do
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=localhost:12392 E18_NPROC=2 E18_PID=$pid \
  python driver.py --config e18-rehearsal.yaml --skip-preflight-check \
  > /tmp/e18-resume-$pid.log 2>&1 &
done
wait
grep -c "skip" /tmp/e18-resume-0.log
echo "L0 REHEARSAL PASS"
