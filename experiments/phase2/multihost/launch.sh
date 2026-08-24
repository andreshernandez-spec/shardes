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

# 2x8 transport selection. Try IB (the fast fabric). If it fails, probe the
# socket transport: that doubles as a BAD-HOST detector, because the recurring
# failing host segfaults on BOTH transports. socket-also-fails => bad host/pair,
# abort fast for a fresh one (do NOT spend the IB-debug hour on a dead node).
# socket-works => a safety net exists, so spend up to 1h (Andres-authorised)
# trying the known NCCL RETRY_EXC fixes to recover IB, else run over socket.
# The winning transport is exported and folded into IFENV for every later step.
phase "preflight 2x8 (transport select: IB, socket bad-host gate, IB debug)"
TRANSPORT=""; IBWIN=""; port=$COORD_PORT
if run_2x8_preflight "$port" ""; then
  TRANSPORT="ib"
else
  port=$((port + 10))
  echo "IB default failed; probing socket over ${NCCL_SOCKET_IFNAME:-overlay} (bad-host gate)"
  if run_2x8_preflight "$port" "NCCL_IB_DISABLE=1 NCCL_NET=Socket"; then
    echo "socket works: host is good, IB-specific problem"
  else
    echo "socket 2x8 ALSO failed: bad host/pair, not IB config; aborting for a fresh cluster"
    exit 1
  fi
  if ib_healthy; then
    echo "spending up to 1h to recover IB (socket stands as the fallback)"
    ib_diag
    DEBUG_DEADLINE=$(( $(date +%s) + 3600 ))
    for combo in "NCCL_IB_GID_INDEX=3" "NCCL_IB_GID_INDEX=1" \
                 "NCCL_IB_GID_INDEX=3 NCCL_IB_TC=106" "NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3"; do
      [ "$(date +%s)" -ge "$DEBUG_DEADLINE" ] && { echo "IB debug budget (1h) spent"; break; }
      port=$((port + 10)); echo ">> IB debug attempt: $combo on port $port"
      if run_2x8_preflight "$port" "$combo"; then TRANSPORT="ib"; IBWIN="$combo"; break; fi
    done
  fi
  if [ "$TRANSPORT" = ib ]; then
    export $IBWIN; IFENV="$IFENV $IBWIN"
  else
    TRANSPORT="socket"; export NCCL_IB_DISABLE=1 NCCL_NET=Socket
    IFENV="$IFENV NCCL_IB_DISABLE=1 NCCL_NET=Socket"
  fi
fi
echo "E18_TRANSPORT=$TRANSPORT${IBWIN:+ ($IBWIN)}"

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
