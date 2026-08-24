#!/bin/bash
# E18 cluster session, driven from the workstation once an Instant
# Cluster exists (docs/10 sections 4 and 5). Bootstraps both nodes at a
# SHA, gives node 0 an ssh key for node 1 over the overlay, starts
# launch.sh detached on node 0, then rsyncs results every few minutes
# until launch.sh prints E18_SESSION_DONE or the hard cap hits, and
# deletes the cluster either way. Run it detached so the cap survives
# this terminal:
#   setsid nohup bash cluster-session.sh <cluster-id> <sha> \
#     > results-e18/session.log 2>&1 < /dev/null &
# Needs RUNPOD_API_KEY and ~/.ssh/id_runpod (the account key).
set -uo pipefail
CLUSTER=${1:?cluster id}; SHA=${2:?commit sha}
CAP_SECONDS=${E18_CAP_SECONDS:-18000}        # 5 h, docs/10 section 5
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=$HOME/.ssh/id_runpod
SSHOPT="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
API=https://api.runpod.io/v2
T0=$(date +%s)
say() { echo "$(date -u +%FT%TZ) $*"; }

pods() {  # prints: rank host port overlay_ip status, one line per pod
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/clusters/$CLUSTER/pods" |
  python3 -c '
import json, sys
for p in json.load(sys.stdin)["pods"]:
    c = p.get("cluster") or {}; d = (p.get("ssh") or {}).get("direct") or {}
    print(c.get("rank"), d.get("host"), d.get("port"), c.get("ip"), p.get("status"))'
}

teardown() {
  say "deleting cluster $CLUSTER"
  curl -sS -o /dev/null -w "delete -> HTTP %{http_code}\n" -X DELETE \
    -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/clusters/$CLUSTER"
}

# -- 1. both pods RUNNING with a direct endpoint and an overlay ip --------
say "waiting for 2 pods of $CLUSTER"
# Only the primary gets a public ssh endpoint; node 1 is reached through
# node 0 over the overlay with a ProxyJump (the key stays on this machine).
for i in $(seq 1 60); do
  P=$(pods) || true
  if [ "$(grep -c ' RUNNING$' <<<"$P")" = 2 ] && ! grep '^0 ' <<<"$P" | grep -q None \
     && ! grep '^1 ' <<<"$P" | awk '{print $4}' | grep -q None; then break; fi
  sleep 20
done
echo "$P"
read -r _ H0 P0 IP0 _ < <(grep '^0 ' <<<"$P")
read -r _ H1 P1 IP1 _ < <(grep '^1 ' <<<"$P")
if [ -z "${H0:-}" ] || [ "$H0" = None ] || [ "${IP0:-None}" = None ] || [ "${IP1:-None}" = None ]; then
  say "pods never became reachable; tearing down"; teardown; exit 1
fi
N0="root@$H0"
ssh0() { ssh $SSHOPT -p "$P0" "$N0" "$@"; }
ssh1() { ssh $SSHOPT -J "$N0:$P0" "root@$IP1" "$@"; }
IFNAME=${E18_IFNAME:-ens1}
for i in $(seq 1 30); do ssh0 true 2>/dev/null && ssh1 true 2>/dev/null && break; sleep 15; done
ssh0 true && ssh1 true || { say "ssh never came up; tearing down"; teardown; exit 1; }

# -- 2. bootstrap both nodes, in parallel --------------------------------
BRANCH=${E18_BRANCH:-e18-runpod-cluster}
BOOT="set -e; cd /root && rm -rf shardes && git clone -q --depth 50 --branch $BRANCH https://github.com/andreshernandez-spec/shardes.git \
  && cd shardes && git checkout -q $SHA \
  && python3 -m venv .venv && . .venv/bin/activate && pip install -q -e . --no-deps \
  && pip install -q 'jax[cuda12]>=0.11' numpy scipy pyyaml 2>&1 | grep -iv warning | tail -2; \
  python -c 'import jax,socket; print(socket.gethostname(), jax.__version__, len(jax.devices()), jax.devices()[0].device_kind)'"
say "bootstrapping node 0 ($H0:$P0, overlay $IP0) and node 1 (via node 0, overlay $IP1) at $SHA"
ssh0 "$BOOT" > /tmp/e18-boot0.log 2>&1 & b0=$!
ssh1 "$BOOT" > /tmp/e18-boot1.log 2>&1 & b1=$!
wait $b0; r0=$?; wait $b1; r1=$?
cat /tmp/e18-boot0.log /tmp/e18-boot1.log
if [ $r0 -ne 0 ] || [ $r1 -ne 0 ]; then say "bootstrap failed ($r0/$r1); tearing down"; teardown; exit 1; fi

# -- 3. node 0 -> node 1 over the overlay --------------------------------
PUB=$(ssh0 "test -f /root/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/id_ed25519; cat /root/.ssh/id_ed25519.pub")
ssh1 "mkdir -p /root/.ssh && echo '$PUB' >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
ssh0 "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 $IP1 hostname" \
  || { say "node 0 cannot ssh node 1 over the overlay ($IP1); tearing down"; teardown; exit 1; }
say "overlay ssh ok"

# -- 4. launch, detached on node 0 ----------------------------------------
# timeout-bound: the setsid job is detached and keeps running even if the ssh
# channel refuses to close (it hung 37 min once), so kill the client after 90s
# and let the harvest loop pick the run up from e18.pid / e18.log.
timeout 90 ssh $SSHOPT -p "$P0" "$N0" "cd /root/shardes/experiments/phase2/multihost && . /root/shardes/.venv/bin/activate \
  && NODE1=$IP1 E18_NODE0_IP=$IP0 SHA=$SHA E18B=${E18B:-0} NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME setsid bash -c 'echo \$\$ > /root/e18.pid; exec bash launch.sh' > /root/e18.log 2>&1 < /dev/null & sleep 1; echo launched pid \$(cat /root/e18.pid)" \
  || say "launch ssh returned nonzero/timed out; launch.sh is detached, continuing to harvest"
say "launch.sh started on node 0; log /root/e18.log"

# -- 5. harvest loop until done or cap -----------------------------------
mkdir -p "$HERE/results-e18"
while true; do
  sleep 180
  rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/shardes/experiments/phase2/multihost/results-e18/" "$HERE/results-e18/" 2>/dev/null
  rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/shardes/experiments/phase2/multihost/results-e18b/" "$HERE/results-e18b/" 2>/dev/null
  # The preflight's contraction cells land outside multihost/ and were lost with the
  # 2026-08-24 pod: predict.py had used them, so C reached predictions.json and the
  # per-cell records (allreduce_insitu, shard_ratio) did not reach anywhere.
  rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/shardes/experiments/phase2/results-contraction/" "$HERE/../results-contraction/" 2>/dev/null
  rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/e18.log" "$HERE/results-e18/e18.log" 2>/dev/null
  last=$(tail -n 1 "$HERE/results-e18/e18.log" 2>/dev/null)
  say "cells $(ls "$HERE/results-e18"/arm=*.json 2>/dev/null | wc -l); last: $last"
  if grep -q 'E18_SESSION_DONE' "$HERE/results-e18/e18.log" 2>/dev/null; then say "session done"; rc=0; break; fi
  # a wedged host (CUDA_ERROR_UNKNOWN, ssh hangs) is the common failure; a
  # bounded liveness probe detects it instead of blocking the loop forever.
  if ! timeout 40 ssh $SSHOPT -p "$P0" "$N0" 'kill -0 $(cat /root/e18.pid) 2>/dev/null' 2>/dev/null; then
    if timeout 40 ssh $SSHOPT -p "$P0" "$N0" true 2>/dev/null; then say "launch.sh is no longer running (see e18.log)"
    else say "node 0 is unreachable (wedged host); aborting"; fi
    rc=3; break
  fi
  if [ $(( $(date +%s) - T0 )) -ge "$CAP_SECONDS" ]; then say "HARD CAP reached"; rc=3; break; fi
done
rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/shardes/experiments/phase2/multihost/results-e18/" "$HERE/results-e18/" 2>/dev/null
rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/shardes/experiments/phase2/multihost/results-e18b/" "$HERE/results-e18b/" 2>/dev/null
rsync -az -e "ssh $SSHOPT -p $P0" "$N0:/root/e18.log" "$HERE/results-e18/e18.log" 2>/dev/null
teardown
say "uptime $(( ($(date +%s) - T0) / 60 )) min; results in $HERE/results-e18"
exit ${rc:-3}
