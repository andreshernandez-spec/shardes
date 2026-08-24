#!/bin/bash
# E18 cluster session orchestration. Run ON NODE 0 of the rented cluster,
# with node 1 reachable over ssh as $NODE1 (and the repo cloned at the same
# SHA on both nodes at the same path). Every phase can abort cheaply; the
# order enforces docs/10 section 4 L2-L3.
#
#   NODE1=<host-or-ip> SHA=<commit> bash launch.sh
#
# Timeouts: the distributed init is the classic first-contact hang, so both
# preflight and driver invocations are wrapped in `timeout`. If preflight
# 2x8 times out, check the coordinator port (12377) is reachable from node 1
# and NCCL_SOCKET_IFNAME; those two cover most first failures.
set -e
cd "$(dirname "$0")"
COORD_PORT=12377
# The coordinator address node 1 dials. On an Instant Cluster pass the
# overlay ip (E18_NODE0_IP); hostname -I's first address is the fallback.
ME=${E18_NODE0_IP:-$(hostname -I | awk '{print $1}')}
: "${NODE1:?set NODE1 to the address of node 1}"
echo "node 0 coordinator $ME:$COORD_PORT, node 1 $NODE1, $(date -u +%FT%TZ)"

# Pin the collectives to the overlay interface: on an Instant Cluster both
# nodes also carry the same 172.18.0.2 bridge address, which NCCL must not pick.
IFENV="NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

run_node1() {  # mirror a command on node 1, same directory, venv and interface pin
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE1" \
    "cd $(pwd) && . /root/shardes/.venv/bin/activate && $IFENV $*"
}

phase() { echo; echo "==== $1 ===="; }

phase "preflight 1x8 (node 0 alone: the anchor, writes the invariance ref)"
E18_TOPOLOGY=1x8 E18_EXPECT_DEVICES=8 timeout 900 python preflight.py

# 2x8 preflight, doubling as the inter-node transport probe. RunPod Instant
# Cluster IB has failed NCCL with RETRY_EXC (2026-08-24, cluster dmzn...); try
# IB first (the fast fabric, the top inter-node point), and if the cross-node
# handshake fails fall back to the socket transport over the overlay, which the
# coordinator and ssh already use. Decided once, then exported so every later
# node-0 command and run_node1 call (via IFENV) inherits the choice. Invoked in
# an `if` so a failed attempt does not trip set -e.
run_2x8_preflight() {  # $1=coordinator port  $2=extra env applied to both nodes
  run_node1 "$2 E18_TOPOLOGY=2x8 E18_COORD=$ME:$1 E18_NPROC=2 E18_PID=1 \
    E18_EXPECT_DEVICES=16 timeout 420 python preflight.py" &
  local n1=$!
  $2 E18_TOPOLOGY=2x8 E18_COORD=$ME:$1 E18_NPROC=2 E18_PID=0 \
    E18_EXPECT_DEVICES=16 timeout 420 python preflight.py
  local rc=$?; wait "$n1" 2>/dev/null || true; return $rc
}

ib_healthy() {  # both nodes still answer and their GPUs are queryable
  timeout 25 nvidia-smi -L >/dev/null 2>&1 && run_node1 "timeout 20 nvidia-smi -L >/dev/null 2>&1"
}

ib_diag() {  # best-effort fabric snapshot to the log; never fatal
  echo "--- IB diag node 0:"; { ibstat; echo '[gids]'; show_gids; ibv_devinfo 2>&1 | grep -iE 'hca_id|state|link_layer|GID'; } 2>&1 | head -40 || true
  echo "--- IB diag node 1:"; run_node1 "{ ibstat; echo '[gids]'; show_gids; ibv_devinfo 2>&1 | grep -iE 'hca_id|state|link_layer|GID'; } 2>&1 | head -40" 2>&1 || true
  return 0
}

# 2x8 preflight = the inter-node transport probe. Try IB first (the fast fabric,
# the top inter-node point). On a clean IB failure Andres has authorised up to
# one hour of uptime to repair it (2026-08-24): dump the fabric and walk the
# known NCCL RETRY_EXC fixes (RoCE GID index, HCA, traffic class) before giving
# up. A wedged host is NOT debugged, it is left to the harvest loop to retry.
# The winning transport is exported and folded into IFENV for every later step.
IB_COMBOS=("" "NCCL_IB_GID_INDEX=3" "NCCL_IB_GID_INDEX=1" \
  "NCCL_IB_GID_INDEX=3 NCCL_IB_TC=106" "NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3" \
  "NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3")
DEBUG_DEADLINE=$(( $(date +%s) + 3600 ))
TRANSPORT=""; IBWIN=""; port=$COORD_PORT

phase "preflight 2x8 (both nodes; IB with up-to-1h debug, then socket)"
for i in "${!IB_COMBOS[@]}"; do
  if [ "$i" -eq 1 ]; then
    echo "IB default failed; entering IB debug (deadline +1h)"
    ib_healthy || { echo "host not healthy after IB failure; skipping debug, will fall back"; break; }
    ib_diag
  fi
  if [ "$i" -gt 0 ] && [ "$(date +%s)" -ge "$DEBUG_DEADLINE" ]; then
    echo "IB debug budget (1h) spent; falling back"; break
  fi
  echo ">> IB attempt [$i]: ${IB_COMBOS[$i]:-default} on port $port"
  if run_2x8_preflight "$port" "${IB_COMBOS[$i]}"; then TRANSPORT="ib"; IBWIN="${IB_COMBOS[$i]}"; break; fi
  port=$((port + 10))
done

if [ "$TRANSPORT" = ib ]; then
  [ -n "$IBWIN" ] && { export $IBWIN; IFENV="$IFENV $IBWIN"; }
  echo "E18_TRANSPORT=ib${IBWIN:+ ($IBWIN)}"
else
  echo "IB unusable; falling back to socket over ${NCCL_SOCKET_IFNAME:-overlay}"
  export NCCL_IB_DISABLE=1 NCCL_NET=Socket
  IFENV="$IFENV NCCL_IB_DISABLE=1 NCCL_NET=Socket"
  sleep 5
  run_2x8_preflight "$((port + 10))" "" || { echo "socket 2x8 also failed; aborting"; exit 1; }
  echo "E18_TRANSPORT=socket"
fi

phase "preflight 2x4 (4 GPUs per node: H1's controlled topology)"
run_node1 "CUDA_VISIBLE_DEVICES=0,1,2,3 E18_TOPOLOGY=2x4 \
  E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=1 E18_EXPECT_DEVICES=8 \
  timeout 900 python preflight.py" &
CUDA_VISIBLE_DEVICES=0,1,2,3 E18_TOPOLOGY=2x4 E18_COORD=$ME:$COORD_PORT \
  E18_NPROC=2 E18_PID=0 E18_EXPECT_DEVICES=8 timeout 900 python preflight.py
wait

phase "contraction isolation on the 1x8 mesh (C for H4's point prediction)"
# Node 0 alone, single process: C is the REPLICATED contraction, the same work on every
# device, so it needs no boundary and no second node. Non-fatal: without it predict.py
# falls back to the coarse bracket it used before C existed, which is a worse prediction
# and not a stopped campaign. Under 5 minutes at these cells.
timeout 1800 python ../contraction_isolation.py \
  --strategies seed_regenerated mirrored_lr1 || \
  echo "WARNING: contraction isolation failed; predictions fall back to the bracket"

phase "predictions (H4): MUST precede any 2x8 campaign cell"
python predict.py

phase "campaign 1x8"
E18_TOPOLOGY=1x8 timeout 3600 python driver.py --config e18.yaml

phase "campaign 2x4"
run_node1 "CUDA_VISIBLE_DEVICES=0,1,2,3 E18_TOPOLOGY=2x4 \
  E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=1 \
  timeout 3600 python driver.py --config e18.yaml" &
CUDA_VISIBLE_DEVICES=0,1,2,3 E18_TOPOLOGY=2x4 E18_COORD=$ME:$COORD_PORT \
  E18_NPROC=2 E18_PID=0 timeout 3600 python driver.py --config e18.yaml
wait

phase "campaign 2x8"
run_node1 "E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=1 \
  timeout 3600 python driver.py --config e18.yaml" &
E18_TOPOLOGY=2x8 E18_COORD=$ME:$COORD_PORT E18_NPROC=2 E18_PID=0 \
  timeout 3600 python driver.py --config e18.yaml
wait

if [ "${E18B:-0}" = 1 ]; then
  phase "E18b: fabric throttle sweep (same cluster, after the baseline)"
  # best-effort: the baseline campaign above is already harvested, so a sweep
  # failure must not abort the session or trigger a wasteful re-acquisition.
  NODE1=$NODE1 E18_NODE0_IP=$ME bash launch-e18b.sh || echo "E18b phase failed; baseline stands"
fi

phase "DONE"
echo "E18_SESSION_DONE"
