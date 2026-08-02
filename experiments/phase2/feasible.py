#!/usr/bin/env python
"""Will this sweep fit in memory? Answered before the node is booked.

    python feasible.py --config sweep.yaml --hbm 80

Strategy A regenerates **all N members on every device**, so its perturbation storage is
`N * |params|` and does not fall with `D` at all. Strategy B holds only its own shard, so
`N/D * |params|`. Neither path chunks: `estimator.py` has a chunk knob but `contraction.py`
does not, so this is the real ceiling rather than a pessimistic bound.

That asymmetry is easy to miss and expensive to find on rented hardware. The first version of
`sweep.yaml` asked for six configurations needing **96 GB per device**, including
`d=2048, N=1024, D=1`, which is the `T_1` baseline that the parallel-efficiency curve divides
by. It would have produced a sweep with a hole exactly where the headline number goes.

`docs/03` M6 asks where "A/B storage becomes the binding constraint". This is that number,
and it is worth reporting rather than merely avoiding: it is the concrete argument for seed
regeneration inside the low-rank path.
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
GB = 1024**3

#: Six (d, d) float32 matrices. `shardes.problems.transformer_block`.
def params_bytes(d_model: int) -> int:
    return 6 * d_model * d_model * 4


def per_device_bytes(d_model: int, population: int, devices: int, how: str) -> int:
    """Perturbation storage on one device.

    A: every device regenerates the whole population, so `D` does not divide anything.
    B: each device holds its own shard.
    """
    members = population if how == "A" else max(1, population // devices)
    return members * params_bytes(d_model)


def populations(cfg: dict, mode: str, d_model: int, devices: int) -> list[int]:
    """Per-model-size populations if given as a mapping, else one list for all sizes.

    **Keys are matched by value, not by type.** YAML parses `512:` as an int, but a config
    written out as JSON (which YAML also parses) comes back with `"512"`, and a hand-written
    config may quote the key. Indexing with the int alone raised `KeyError: 512` inside a
    Kaggle kernel, after the install and the clone, which is an expensive place to learn it.
    """
    key = "population" if mode == "strong" else "population_per_device"
    spec = cfg[key]
    if isinstance(spec, dict):
        by_value = {int(k): v for k, v in spec.items()}
        if d_model not in by_value:
            raise KeyError(
                f"{key} has no entry for d_model={d_model}; it has {sorted(by_value)}"
            )
        values = by_value[d_model]
    else:
        values = spec
    return [int(n) if mode == "strong" else int(n) * devices for n in values]


def audit(cfg: dict, hbm_gb: float, headroom: float = 0.9) -> list[tuple]:
    budget = hbm_gb * GB * headroom
    rows = []
    for mode in cfg["modes"]:
        for d_model in cfg["d_model"]:
            for devices in cfg["devices"]:
                for population in populations(cfg, mode, d_model, devices):
                    for how in cfg["how"]:
                        need = per_device_bytes(d_model, population, devices, how)
                        rows.append((need / GB, need > budget, mode, d_model,
                                     population, devices, how))
    rows.sort(reverse=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=pathlib.Path, default=HERE / "sweep.yaml")
    ap.add_argument("--hbm", type=float, default=80.0, help="per-device HBM, GB")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    rows = audit(cfg, args.hbm)
    over = [r for r in rows if r[1]]

    print(f"{args.config.name}, {args.hbm:g} GB per device, 90% headroom\n")
    print(f"{'GB/dev':>8} {'mode':7} {'d':>5} {'N':>5} {'D':>3} {'how':>4}")
    for gb, bad, mode, d, n, dev, how in rows[: args.show]:
        print(f"{gb:>8.1f} {mode:7} {d:>5} {n:>5} {dev:>3} {how:>4}"
              f"{'   << over budget' if bad else ''}")

    print(f"\n{len(rows)} configurations, {len(over)} over budget")
    if over:
        print("\nThese would be recorded as errors and leave holes in the curves. The one to")
        print("check first is D=1, which every parallel-efficiency figure divides by:")
        for gb, _, mode, d, n, dev, how in over:
            if dev == 1:
                print(f"  MISSING BASELINE  {mode} d={d} N={n} how={how}  needs {gb:.0f} GB")
        return 1
    print("every configuration fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
