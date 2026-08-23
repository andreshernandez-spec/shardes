#!/bin/bash
# E18b: the fabric throttle sweep, run ON NODE 0 after the baseline campaign,
# on the SAME rented cluster (docs/10 follow-on). Forces NCCL onto the socket
# transport so tc on the overlay actually throttles collectives, sweeps the
# overlay bandwidth, and re-measures beta at each setting. Predictions are
# frozen after the socket-native calibration and before the throttled cells.
#
#   NODE1=<node1-overlay-ip> E18_NODE0_IP=<node0-overlay-ip> bash launch-e18b.sh
#
# Requires the baseline to have run first (it writes the invariance reference
# and the 1x8 ladder that predict_e18b reads). Degrades to the socket-native
# point alone if tc is unusable in the container.
set -e
cd "$(dirname "$0")"
COORD_PORT=${E18B_COORD_PORT:-12388}
ME=${E18_NODE0_IP:-$(hostname -I | awk '{print $1}')}
: "${NODE1:?set NODE1 to the node 1 overlay ip}"
IFACE=$(bash throttle.sh iface)
export E18_RESULTS_DIR=results-e18b
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE
NCCLENV="NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE"
echo "E18b: overlay iface $IFACE, coordinator $ME:$COORD_PORT, socket transport forced"

run1() { ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=2 "$NODE1" "cd $(pwd) && . /root/shardes/.venv/bin/activate && $NCCLENV $*"; }
phase() { echo; echo "==== $1 ($(date -u +%FT%TZ)) ===="; }

# Capability gate: tc needs NET_ADMIN and the tbf module in the container. If
# either node cannot tc, keep only the socket-native point (needs no tc).
SETTINGS="socket-native rate-10gbit rate-1gbit"
if ! bash throttle.sh probe >/dev/null 2>&1 || ! run1 "bash throttle.sh probe" >/dev/null 2>&1; then
  echo "E18b: tc unavailable on a node, keeping socket-native only"
  SETTINGS="socket-native"
fi
echo "E18b settings: $SETTINGS"

rate_of() { case $1 in socket-native) echo none;; rate-10gbit) echo 10gbit;; rate-1gbit) echo 1gbit;; esac; }

first=1
for s in $SETTINGS; do
  RATE=$(rate_of "$s")
  phase "setting $s (rate $RATE)"
  if [ "$RATE" = none ]; then bash throttle.sh clear; run1 "bash throttle.sh clear";
  else bash throttle.sh set "$RATE"; run1 "bash throttle.sh set $RATE"; fi

  phase "preflight 2x8 @ $s (measures beta over the throttled socket)"
  run1 "E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=1 E18_EXPECT_DEVICES=16 E18B_SETTING=$s timeout 900 python preflight.py" &
  E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=0 E18_EXPECT_DEVICES=16 E18B_SETTING=$s timeout 900 python preflight.py
  wait

  if [ "$first" = 1 ]; then
    phase "predictions (G4): frozen from socket-native + committed sweep, before throttled cells"
    python predict_e18b.py
    first=0
  fi

  phase "campaign 2x8 @ $s"
  run1 "E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=1 E18B_SETTING=$s timeout 3600 python driver.py --config e18b.yaml" &
  E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=0 E18B_SETTING=$s timeout 3600 python driver.py --config e18b.yaml
  wait
done

bash throttle.sh clear; run1 "bash throttle.sh clear" || true
phase "DONE"
echo "E18B_DONE"
