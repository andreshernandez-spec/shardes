#!/usr/bin/env python
"""E18 campaign driver: the contraction crossover across a host boundary.

One program, every topology. The launcher decides the topology by how it
starts processes; the driver reads it from the environment and stamps it
into every record:

    E18_TOPOLOGY  label for the records ("1x8", "2x4", "2x8")
    E18_COORD     coordinator host:port  (unset => single-process)
    E18_NPROC     number of processes    (unset => 1)
    E18_PID       this process's id

    # single process, 8 devices (the anchor topology):
    E18_TOPOLOGY=1x8 python driver.py --config e18.yaml
    # two processes (run once per node, or twice locally for rehearsal):
    E18_TOPOLOGY=2x8 E18_COORD=node0:12377 E18_NPROC=2 E18_PID=<0|1> \
        python driver.py --config e18.yaml

Process 0 is the only writer. One JSON per cell, written when the cell
finishes, so a killed session resumes by skipping existing files. The
driver refuses to run the campaign unless preflight.py's pass marker for
this topology exists (--skip-preflight-check exists for the CPU rehearsal).

Requires that predictions.json exists before any 2x8 cell runs when the
config names frozen predictions; preflight and predict.py produce it. The
enforcement is deliberate: the prediction must be on disk before the cells
it predicts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # experiments/ for harness
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

import jax  # noqa: E402

NPROC = int(os.environ.get("E18_NPROC", "1"))
PID = int(os.environ.get("E18_PID", "0"))
TOPOLOGY = os.environ.get("E18_TOPOLOGY", "unset")

if NPROC > 1:
    jax.distributed.initialize(coordinator_address=os.environ["E18_COORD"],
                               num_processes=NPROC, process_id=PID)

import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402

import harness  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

STRATEGIES = {
    "seed_regenerated": lambda: SeedRegenerated(),
    "mirrored_lr1": lambda: Mirrored(LowRank(r=1)),
}


def log(msg: str) -> None:
    if PID == 0:
        print(msg, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--skip-preflight-check", action="store_true",
                    help="CPU rehearsal only; the cluster session never passes this")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])

    out = HERE / cfg["results_dir"]
    if PID == 0:
        out.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight_check:
        marker = out / f"preflight-{TOPOLOGY}.json"
        if not marker.exists():
            log(f"REFUSING to run: no preflight pass marker {marker.name}. "
                "Run preflight.py for this topology first.")
            return 2

    n_devices = len(jax.devices())
    mesh = sharding.make_mesh(n_devices)
    env = harness.capture_env(HERE, (cfg["results_dir"],)) if PID == 0 else None
    log(f"topology {TOPOLOGY}: {NPROC} process(es), {n_devices} global devices")

    for arm in cfg["arms"]:
        for how in cfg["hows"]:
            for cell in cfg["cells"]:
                d_model, n = cell["d"], cell["N"]
                name = (f"arm={arm}__how={how}__d={d_model}__N={n}"
                        f"__topo={TOPOLOGY}.json")
                outfile = out / name
                if outfile.exists():
                    log(f"skip {name}: already measured")
                    continue
                key = jax.random.key(0)
                params = transformer_block.init(key, d_model=d_model)
                batch = transformer_block.make_batch(
                    jax.random.fold_in(key, 1), d_model=d_model,
                    batch=cfg["batch"], seq=cfg["seq"])
                es = ShardedES(STRATEGIES[arm](), n=n, sigma=0.01, lr=0.05,
                               mesh=mesh, how=how)
                state = es.init(key, params)

                @jax.jit
                def generation(state):
                    pert, state = es.ask(state)
                    fitness = es.apply(transformer_block.loss, state,
                                       pert)(batch)
                    return es.tell(state, pert, fitness)

                for _ in range(cfg["warmup"]):
                    state = generation(state)
                jax.block_until_ready(state.params)
                seconds = []
                for _ in range(cfg["repeats"]):
                    t0 = time.perf_counter()
                    state = generation(state)
                    jax.block_until_ready(state.params)
                    seconds.append(time.perf_counter() - t0)
                if PID == 0:
                    rec = {"config": {"arm": arm, "how": how,
                                      "d_model": d_model, "population": n,
                                      "devices": n_devices,
                                      "topology": TOPOLOGY,
                                      "n_processes": NPROC,
                                      "batch": cfg["batch"],
                                      "seq": cfg["seq"],
                                      "matmul_precision": cfg["matmul_precision"]},
                           "seconds_all": seconds,
                           "seconds_median": statistics.median(seconds),
                           "env": env}
                    outfile.write_text(json.dumps(rec, indent=1))
                log(f"{arm} how={how} d={d_model} N={n} topo={TOPOLOGY}: "
                    f"{statistics.median(seconds) * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
