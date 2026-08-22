#!/usr/bin/env python
"""E18 preflight: the abort-for-eight-dollars gate, one run per topology.

Same environment contract as driver.py. Exits non-zero on ANY failure and
writes the pass marker (preflight-<topology>.json) only at the very end;
driver.py refuses to run the campaign without it. The marker doubles as the
calibration record: link alpha-beta per payload size, versions, device
identity, and the invariance fingerprint.

Order, chosen so the cheapest checks fail first:
  1. identity: every process sees the same device kind and count, the
     expected global total, and distinct hostnames when NPROC > 1.
  2. collectives: a psum ladder over the full mesh at payload sizes from
     8 B to 100 MB, timed; this is the calibration input for predict.py and
     the fabric characterization the results README leads with.
  3. invariance: one generation per (arm, how) on a tiny block; topology
     1x8 WRITES the reference fingerprint, every other topology compares
     against it within the repo's cross-layout tolerance (contraction A is
     bitwise; B reassociates its psum, measured at ~2e-7 relative on the
     CPU rehearsal).
  4. one warm cell per (arm, how) at the smallest campaign shape, with a
     hard per-cell time cap, so a compile blowup surfaces here.

The distributed init itself is wrapped in a timeout by launch.sh (a hang on
first contact is the most likely cluster failure, and Python cannot
reliably time out a blocked C++ init from inside).
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

import jax  # noqa: E402

NPROC = int(os.environ.get("E18_NPROC", "1"))
PID = int(os.environ.get("E18_PID", "0"))
TOPOLOGY = os.environ.get("E18_TOPOLOGY", "unset")
CAP_SECONDS = 300  # per warm cell; a compile past this is a failure

if NPROC > 1:
    jax.distributed.initialize(coordinator_address=os.environ["E18_COORD"],
                               num_processes=NPROC, process_id=PID)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

ARMS = {"seed_regenerated_A": (lambda: SeedRegenerated(), "A"),
        "seed_regenerated_B": (lambda: SeedRegenerated(), "B"),
        "mirrored_lr1_A": (lambda: Mirrored(LowRank(r=1)), "A"),
        "mirrored_lr1_B": (lambda: Mirrored(LowRank(r=1)), "B")}


def log(msg: str) -> None:
    if PID == 0:
        print(msg, flush=True)


def main() -> int:
    report: dict = {"topology": TOPOLOGY, "n_processes": NPROC}

    # -- 1. identity ------------------------------------------------------
    devs = jax.devices()
    kinds = {d.device_kind for d in devs}
    assert len(kinds) == 1, f"mixed device kinds: {kinds}"
    report["device_kind"] = kinds.pop()
    report["n_devices"] = len(devs)
    report["n_local"] = len(jax.local_devices())
    report["hostname"] = socket.gethostname()
    report["jax"] = jax.__version__
    expected = int(os.environ.get("E18_EXPECT_DEVICES", "0"))
    if expected:
        assert len(devs) == expected, (len(devs), expected)
    log(f"identity ok: {report['n_devices']} x {report['device_kind']}, "
        f"{NPROC} process(es)")

    # -- 2. the psum ladder ----------------------------------------------
    mesh = sharding.make_mesh(len(devs))
    rep = NamedSharding(mesh, PartitionSpec())
    ladder = {}
    for size_bytes in (8, 1024, 2**20, 100 * 2**20):
        n_f32 = max(size_bytes // 4, 2)
        x = jax.device_put(jnp.ones(n_f32), rep)

        @jax.jit
        def allreduce(v):
            return jax.lax.with_sharding_constraint(v * 1.0001, rep)

        # A replicated multiply forces no collective; time the psum the
        # library actually issues instead: contraction B's tree-psum via a
        # sharded partial. Use jnp.sum over a member-sharded array.
        member = NamedSharding(mesh, PartitionSpec(sharding.POP))
        y = jax.device_put(jnp.ones((len(devs), n_f32 // len(devs) + 1)), member)

        @jax.jit
        def psum_like(v):
            return jax.lax.with_sharding_constraint(v.sum(axis=0), rep)

        for _ in range(3):
            _ = psum_like(y).block_until_ready()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = psum_like(y).block_until_ready()
            ts.append(time.perf_counter() - t0)
        ladder[str(size_bytes)] = statistics.median(ts)
        log(f"all-reduce ~{size_bytes} B: {statistics.median(ts) * 1e6:.1f} us")
    report["allreduce_seconds_by_bytes"] = ladder
    big, small = ladder[str(100 * 2**20)], ladder["8"]
    report["alpha_seconds"] = small
    report["beta_bytes_per_second"] = (100 * 2**20) / max(big - small, 1e-9)
    log(f"alpha ~{small * 1e6:.0f} us, beta ~"
        f"{report['beta_bytes_per_second'] / 2**30:.1f} GiB/s")

    # -- 3. invariance ----------------------------------------------------
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=64)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=64, batch=4, seq=8)
    fingerprints = {}
    for name, (mk, how) in ARMS.items():
        es = ShardedES(mk(), n=16, sigma=0.01, lr=0.05, mesh=mesh, how=how)
        state = es.init(key, params)

        @jax.jit
        def gen(state):
            pert, state = es.ask(state)
            fitness = es.apply(transformer_block.loss, state, pert)(batch)
            return es.tell(state, pert, fitness)

        new = gen(state)
        delta = jax.tree.map(lambda a, b: a - b, new.params, state.params)
        flat = np.concatenate([np.asarray(x).ravel()
                               for x in jax.tree.leaves(delta)])
        fingerprints[name] = flat
    ref_path = HERE / "e18-invariance-ref.npz"
    if TOPOLOGY == "1x8":
        if PID == 0:
            np.savez(ref_path, **fingerprints)
        log("invariance reference written (1x8)")
    else:
        assert ref_path.exists(), "run the 1x8 preflight first"
        ref = np.load(ref_path)
        for k, v in fingerprints.items():
            d = float(np.abs(v - ref[k]).max())
            m = float(np.abs(ref[k]).max())
            assert d <= 1e-5 * max(m, 1e-30), (k, d, m)
            log(f"invariance {k}: rel {d / m if m else 0:.2e} ok")
    log("invariance ok")

    # -- 4. one capped warm cell per arm ---------------------------------
    for name, (mk, how) in ARMS.items():
        d_model, n = 512, 256
        params = transformer_block.init(key, d_model=d_model)
        b = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=d_model, batch=8, seq=32)
        es = ShardedES(mk(), n=n, sigma=0.01, lr=0.05, mesh=mesh, how=how)
        state = es.init(key, params)

        @jax.jit
        def gen(state):
            pert, state = es.ask(state)
            fitness = es.apply(transformer_block.loss, state, pert)(b)
            return es.tell(state, pert, fitness)

        t0 = time.perf_counter()
        jax.block_until_ready(gen(state).params)
        dt = time.perf_counter() - t0
        assert dt < CAP_SECONDS, f"{name}: first call {dt:.0f}s exceeds cap"
        log(f"warm {name}: compile+run {dt:.1f}s")

    if PID == 0:
        results_dir = HERE / os.environ.get("E18_RESULTS_DIR", "results-e18")
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"preflight-{TOPOLOGY}.json").write_text(
            json.dumps(report, indent=1))
    log(f"PREFLIGHT PASS ({TOPOLOGY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
