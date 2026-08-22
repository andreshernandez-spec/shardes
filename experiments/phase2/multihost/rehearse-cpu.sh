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

echo "== preflight 2x4 (two processes, 4 simulated devices each) =="
for pid in 0 1; do
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=$COORD E18_NPROC=2 E18_PID=$pid \
  E18_EXPECT_DEVICES=8 E18_RESULTS_DIR=results-e18-rehearsal \
  python preflight.py > /tmp/e18-preflight-$pid.log 2>&1 &
done
wait
grep "PREFLIGHT PASS" /tmp/e18-preflight-0.log

echo "== driver 1x8, tiny config =="
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  E18_TOPOLOGY=1x8 python driver.py --config e18-rehearsal.yaml \
  --skip-preflight-check

echo "== driver 2x4, tiny config (resume check: rerun skips) =="
for pid in 0 1; do
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  E18_TOPOLOGY=2x4 E18_COORD=localhost:12391 E18_NPROC=2 E18_PID=$pid \
  python driver.py --config e18-rehearsal.yaml --skip-preflight-check \
  > /tmp/e18-driver-$pid.log 2>&1 &
done
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
