# 13 - E18 on Crusoe: the full range RunPod could not give

Written 2026-08-24 after the RunPod run (experiments/phase2/multihost/results-e18).
RunPod gave a clean two-point result (NVLink 515 GiB/s, socket 9.2 GiB/s) but
cannot give the other two things E18/E18b wanted, and neither is a RunPod config
we missed:

- **InfiniBand.** Present on every RunPod host (mlx5) but NCCL fails it with
  RETRY_EXC on all five clusters we drew. Likely a RoCE GID/partition issue their
  platform does not expose; the launcher's IB-recovery matrix did not crack it.
- **The throttle sweep.** `tc` needs NET_ADMIN, RunPod containers do not grant it
  (create API has no capability field; open feature request runpodctl#272), and
  the userspace alternatives (trickle) are unproven with NCCL and risk collective
  timeouts.

Crusoe fixes both, because it rents real VMs, not containers: root with
NET_ADMIN (so `tc` works) and real Quantum-2 InfiniBand.

## Procurement (once quota lands)

Two `a100-80gb-sxm-ib.8x` VMs in `us-east1-a` (the only Crusoe location with A100
IB), one IB partition:

```
crusoe networking ib-partitions create --name e18 --ib-network-id <net>
crusoe compute vms create --name e18-0 --type a100-80gb-sxm-ib.8x \
  --location us-east1-a --ib-partition-id <part> --image ubuntu22.04-nvidia-sxm-docker:latest
crusoe compute vms create --name e18-1 --type a100-80gb-sxm-ib.8x ... (same partition)
```

Cost about $36.80/h for the pair; a full session (IB point + baseline + throttle
sweep) is ~2 h, ~$75. Quota to request: 2 a100-80gb-sxm-ib.8x instances (16 GPUs),
1 IB network, 1 IB partition, 2 public IPs, ~200 GB disk. Nothing else.

## What changes in the harness

The code is done and debugged; only procurement differs. `cluster-session.sh`
takes a RunPod cluster id and derives the two node addresses from it. For Crusoe,
pass the two VM addresses directly. Add a `--nodes <ip0> <ip1>` path to
`cluster-session.sh` (or a thin `crusoe-session.sh`) that sets NODE0/NODE1 from
the VM IPs and skips the RunPod `/v2/clusters` calls; everything downstream
(bootstrap, transport select, launch.sh, launch-e18b.sh, harvest, teardown) is
unchanged. The overlay interface is the IB/VPC nic, not `ens1`; set
`E18_IFNAME`/`E18_NODE0_IP` from `ip -o addr` on the VM.

## What we expect to get

- **The IB point.** On real Crusoe IB the transport select should pick IB (no
  RETRY_EXC), giving a ~100 to 400 Gb/s inter-node point above the socket 9 GiB/s.
- **The throttle sweep.** `tc` works, so E18b runs socket-native, 10 Gbit, 1 Gbit
  as designed, each beta re-measured by the ladder. Watch the 1 Gbit point: a
  100 MB all-reduce at 1 Gbit is ~0.8 s and may hit NCCL's collective timeout;
  raise the timeout or drop the largest ladder payload for that setting.
- **The full range:** NVLink 515 -> IB -> socket 9 -> 10 Gbit -> 1 Gbit, which
  tests H4's model across three-plus orders instead of the two points RunPod gave.

## What already stands without this

The RunPod result (results-e18) is complete on its own: the boundary cliff (H1,
~108 ms), the seed_regenerated B->A crossover, and the H4 miss (the model
under-predicts B's boundary penalty for seed_regenerated at large d). Crusoe
strengthens the H4 curve and adds the IB anchor; it is not required for the core
claim. Fix the harvest first either way: the RunPod run lost results-e18b to an
rsync that only pulled results-e18 (fixed in cluster-session.sh, 2026-08-24).
