#!/bin/bash
# RunPod Instant Cluster for E18 (docs/10 section 5), from the REST v2 API.
# The curated MCP tools do not expose /v2/clusters; this does, with curl.
#
#   RUNPOD_API_KEY=... bash runpod-cluster.sh <command> [args]
#
#   try              one create attempt; prints the cluster id on success
#                    (exit 0), the API error on failure (exit 1). A failed
#                    create costs nothing and is the only stock probe the
#                    API offers for clusters. A successful one BILLS from
#                    that second, so run it only when ready to use it.
#   wait [secs]      repeat `try` every secs (default 300) until it lands.
#   list             clusters on the account
#   status <id>      cluster summary: data center, pod counts, primary ssh
#   pods <id>        per member pod: rank, overlay ip, status, direct ssh
#   delete <id>      terminate; billing stops here, nothing else stops it
#   billing          cluster billing records
#   body             print the create request body without sending it
#   arm <sha> [secs] detached: wait for a cluster, then run
#                    cluster-session.sh <id> <sha>; log results-e18/arm.log
#   session <id> <sha>  detached: cluster-session.sh on an existing cluster
#
# Shape and image come from the environment, defaults are the E18 ones:
#   E18_PODS=2 E18_GPUS_PER_POD=8 E18_GPU="NVIDIA A100-SXM4-80GB"
#   E18_IMAGE=runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster
#   E18_DISK=100 E18_DCS=  (comma list of data center ids, empty = any)
#   E18_SSH_PUBKEY=~/.ssh/id_runpod.pub
# Every pod gets 22/tcp published and PUBLIC_KEY set (the registered
# account key via startSsh too), so the direct host:port works as for pods.
set -euo pipefail

API=https://api.runpod.io/v2
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"
NAME=${E18_NAME:-shardes-e18}
PODS=${E18_PODS:-2}
GPUS_PER_POD=${E18_GPUS_PER_POD:-8}
GPU=${E18_GPU:-NVIDIA A100-SXM4-80GB}
IMAGE=${E18_IMAGE:-runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster}
DISK=${E18_DISK:-100}
DCS=${E18_DCS:-}
PUBKEY_FILE=${E18_SSH_PUBKEY:-$HOME/.ssh/id_runpod.pub}

api() {  # api METHOD PATH [BODY]; prints body, returns 0 on 2xx
  local method=$1 path=$2 body=${3:-} out code
  out=$(curl -sS -X "$method" "$API$path" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    ${body:+--data "$body"} -w '\n%{http_code}')
  code=${out##*$'\n'}
  printf '%s\n' "${out%$'\n'*}"
  [[ $code == 2* ]]
}

create_body() {
  python3 - "$NAME" "$PODS" "$GPUS_PER_POD" "$GPU" "$IMAGE" "$DISK" "$DCS" "$PUBKEY_FILE" <<'PY'
import json, sys
name, pods, per, gpu, image, disk, dcs, pubkey = sys.argv[1:]
body = {
  "name": name, "type": "TRAINING",
  "compute": {"gpuTypeId": gpu, "gpuCountPerPod": int(per), "podCount": int(pods)},
  "image": image, "disk": int(disk), "ports": ["22/tcp"],
  "env": {"PUBLIC_KEY": open(pubkey).read().strip()},
  "startSsh": True,
}
if dcs: body["dataCenterIds"] = dcs.split(",")
print(json.dumps(body))
PY
}

cmd=${1:-}; shift || true
case $cmd in
  try)
    echo "$(date -u +%FT%TZ) create ${PODS}x${GPUS_PER_POD} '$GPU' dcs='${DCS:-any}'" >&2
    if out=$(api POST /clusters "$(create_body)"); then
      python3 -c "import json,sys; c=json.load(sys.stdin); print(c['id'])" <<<"$out"
    else
      echo "$out" >&2; exit 1
    fi ;;
  wait)
    every=${1:-300}
    while true; do
      if id=$(bash "$0" try 2>>"${E18_WAIT_LOG:-/dev/stderr}"); then
        echo "$(date -u +%FT%TZ) CLUSTER $id" >&2; echo "$id"; exit 0
      fi
      sleep "$every"
    done ;;
  list)    api GET /clusters | python3 -m json.tool ;;
  status)  api GET "/clusters/${1:?id}" | python3 -m json.tool ;;
  pods)
    api GET "/clusters/${1:?id}/pods" | python3 -c '
import json, sys
for p in json.load(sys.stdin)["pods"]:
    c = p.get("cluster") or {}
    ssh = p.get("ssh") or {}
    print(p["id"], "rank", c.get("rank"), "ip", c.get("ip"), p.get("status"),
          "dc", p.get("dataCenterId"), "ssh", json.dumps(ssh.get("direct")))' ;;
  delete)  api DELETE "/clusters/${1:?id}" >/dev/null && echo "deleted ${1}" ;;
  billing) api GET /billing/clusters | python3 -m json.tool ;;
  body)    create_body | python3 -m json.tool ;;
  session)
    cid=${1:?cluster id}; sha=${2:?commit sha}
    dir=$(cd "$(dirname "$0")" && pwd); mkdir -p "$dir/results-e18"
    unit=e18-session-$(date -u +%Y%m%dT%H%M%SZ)
    systemd-run --user --unit="$unit" --collect --same-dir \
      --setenv=RUNPOD_API_KEY="$RUNPOD_API_KEY" --setenv=HOME="$HOME" --setenv=PATH="$PATH" \
      --setenv=E18B="${E18B:-0}" \
      bash -c "exec >> '$dir/results-e18/arm.log' 2>&1 < /dev/null; bash '$dir/cluster-session.sh' $cid $sha"
    echo "session unit $unit on $cid at $sha (systemctl --user stop $unit to abandon; the cluster stays)" ;;
  arm)
    sha=${1:?commit sha for the session}; every=${2:-300}
    dir=$(cd "$(dirname "$0")" && pwd); mkdir -p "$dir/results-e18"
    # Loop: acquire a cluster, run a session; a session that ABORTS (bad host,
    # launch died, cap) exits 3, so acquire another. Stop when one COMPLETES
    # (exit 0) or after E18_MAX_ATTEMPTS tries. Each cluster is torn down by
    # cluster-session.sh before the next acquisition.
    max=${E18_MAX_ATTEMPTS:-8}
    chain="for a in \$(seq 1 $max); do echo \"\$(date -u +%FT%TZ) attempt \$a/$max\"; id=\$(bash '$dir/runpod-cluster.sh' wait $every) || break; bash '$dir/cluster-session.sh' \$id $sha && break; done"
    # A transient systemd user unit outlives the shell that started it; a
    # setsid'd process from a tool shell was killed twice (2026-08-23).
    if command -v systemd-run >/dev/null; then
      unit=e18-arm-$(date -u +%Y%m%dT%H%M%SZ)
      systemd-run --user --unit="$unit" --collect --same-dir \
        --setenv=RUNPOD_API_KEY="$RUNPOD_API_KEY" --setenv=HOME="$HOME" --setenv=PATH="$PATH" \
        --setenv=E18B="${E18B:-0}" \
        bash -c "exec >> '$dir/results-e18/arm.log' 2>&1 < /dev/null; $chain"
      echo "armed: unit $unit (journalctl --user -u $unit; systemctl --user stop $unit), sha $sha, try every ${every}s, log $dir/results-e18/arm.log"
    else
      setsid nohup bash -c "$chain" >> "$dir/results-e18/arm.log" 2>&1 < /dev/null &
      echo "armed: pid $!, sha $sha, try every ${every}s, log $dir/results-e18/arm.log"
    fi ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
