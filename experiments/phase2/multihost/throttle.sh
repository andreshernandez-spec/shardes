#!/bin/bash
# tc bandwidth limiting on the cluster overlay interface, for E18b.
#   throttle.sh iface              print the overlay interface (the 10.65.x nic)
#   throttle.sh probe [iface]      exit 0 iff a tbf qdisc can be added+removed here
#   throttle.sh set <rate> [burst] apply egress tbf (e.g. set 1gbit)
#   throttle.sh clear
# The overlay is the cluster VXLAN (10.65.x.x); tbf on its egress on EVERY
# node throttles the collective symmetrically (all-reduce is symmetric). tbf,
# not netem: we cap bandwidth (beta) and leave latency the fabric's own. The
# preflight ladder re-measures beta after each set, so a nominal rate that the
# limiter does not hit exactly is recorded as its measured value, not assumed.
set -uo pipefail
overlay_iface() { ip -o -4 addr show 2>/dev/null | awk '/ 10\.65\./{print $2; exit}'; }
case "${1:-}" in
  iface) echo "$(overlay_iface || true)";;
  probe)
    ifc=${2:-$(overlay_iface || true)}
    [ -n "$ifc" ] || { echo "no interface to probe (no 10.65.x nic?)"; exit 1; }
    if tc qdisc add dev "$ifc" root tbf rate 1gbit burst 2mb latency 400ms 2>/tmp/tcerr; then
      tc qdisc del dev "$ifc" root 2>/dev/null; echo "tc ok on $ifc"; exit 0
    fi
    echo "tc unavailable on $ifc: $(cat /tmp/tcerr)"; exit 1;;
  set)
    rate=${2:?rate}; burst=${3:-2mb}; ifc=$(overlay_iface || true)
    [ -n "$ifc" ] || { echo "no overlay interface"; exit 1; }
    tc qdisc replace dev "$ifc" root tbf rate "$rate" burst "$burst" latency 400ms
    echo "throttled $ifc to $rate (burst $burst)";;
  clear)
    ifc=$(overlay_iface || true)
    [ -n "$ifc" ] && tc qdisc del dev "$ifc" root 2>/dev/null || true
    echo "cleared ${ifc:-none}";;
  *) echo "usage: throttle.sh {iface|probe [iface]|set <rate> [burst]|clear}"; exit 2;;
esac
