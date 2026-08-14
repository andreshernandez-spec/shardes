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


#: Live activation tensors in the batched forward pass, fitted to measurement rather than
#: derived. 2.9 reproduces every low-rank row on a T4 and a 3080 across d and N; 3 is used.
_ACTIVATION_TENSORS = 3
#: `iid_gaussian` holds more than one copy of the perturbation between sample and contract.
#: Measured 1.4x to 2.0x of N*|params|; the larger is used because under-predicting this is
#: what OOMs a rented node.
_MATERIALISE = 2.0


#: Strategies that walk members one at a time rather than batching them. Their activations do
#: not grow with the population, and neither does their perturbation storage. Measured:
#: `seed_regenerated` sits at 0.01 GB (d=512) and 0.15 GB (d=2048) for every N tried, while
#: the batched strategies grow linearly. It pays in wall clock, about 4x the low-rank path at
#: d=2048, N=512, which is the trade worth reporting rather than hiding.
#: Strategies whose evaluation is a loop rather than a batched op, so only one
#: member is live at a time. `mirrored_seed` wraps `seed_regenerated` and inherits
#: its scan, which is why it inherits its memory profile too.
SEQUENTIAL = {"seed_regenerated", "mirrored_seed"}


def activation_bytes(d_model: int, members_here: int, batch: int = 8, seq: int = 32) -> int:
    """Activations for the members this device evaluates. Strategy-independent.

    **Left out of the first model entirely, which is half of why it was wrong.** At
    `d=512, N=256` the low-rank configs measured 0.39 GB while storing about 1 MB of
    perturbation: essentially all of it is the forward pass over members.
    """
    return _ACTIVATION_TENSORS * members_here * batch * seq * d_model * 4


def perturbation_bytes(strategy: str, d_model: int, members_contracted: int) -> int:
    """What the strategy itself stores. **This is strategy-dependent, and assuming it was not
    is the other half of why the first model was wrong.**

    - `iid_gaussian` materialises the population: `N * |params|`, and then some.
    - `seed_regenerated` re-derives members instead of storing them, so this is ~independent
      of `N`. Measured 0.01 GB at d=512 for every N tried, 0.15 GB at d=2048. It buys that
      with time: 4x slower than the low-rank path at d=2048, N=512.
    - `mirrored_seed` is `Mirrored(SeedRegenerated())`, Qiu et al. as published. It wraps the
      same regeneration, so it stores the same couple of buffers: antithetic pairing halves
      the *distinct directions* and not the storage, which was already `O(|params|)`.
    - the low-rank path never materialises an `(m, n)` perturbation at all (invariant 3), so
      it stores `N * r * (m + n)`, which is ~1 MB where the old model predicted 1.5 GB.
    """
    if strategy in ("seed_regenerated", "mirrored_seed"):
        return 2 * params_bytes(d_model)              # a couple of param-sized buffers
    if strategy in ("lowrank_r1", "mirrored_lr1"):
        rank = 1
        return members_contracted * rank * 2 * d_model * 4
    return int(_MATERIALISE * members_contracted * params_bytes(d_model))


def per_device_bytes(d_model: int, population: int, devices: int, how: str,
                     strategy: str = "iid_gaussian", batch: int = 8, seq: int = 32) -> int:
    """Peak per-device bytes, fitted to measurement.

    `how` decides how many members this device *contracts*: A regenerates the whole
    population everywhere, B holds only its shard. Evaluation is sharded either way, so
    activations always scale with `N / D`.

    Defaults to `iid_gaussian` because it is the largest, so a caller that does not care
    about strategy still gets a safe answer.
    """
    contracted = population if how == "A" else max(1, population // devices)
    evaluated = 1 if strategy in SEQUENTIAL else max(1, population // devices)
    return (perturbation_bytes(strategy, d_model, contracted)
            + activation_bytes(d_model, evaluated, batch, seq))


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
                        for strategy in cfg.get("strategies", ["iid_gaussian"]):
                            need = per_device_bytes(d_model, population, devices, how,
                                                    strategy, cfg.get("batch", 8),
                                                    cfg.get("seq", 32))
                            rows.append((need / GB, need > budget, mode, d_model,
                                         population, devices, how, strategy))
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
    print(f"{'GB/dev':>8} {'mode':7} {'d':>5} {'N':>5} {'D':>3} {'how':>4} {'strategy':18}")
    for gb, bad, mode, d, n, dev, how, strat in rows[: args.show]:
        print(f"{gb:>8.1f} {mode:7} {d:>5} {n:>5} {dev:>3} {how:>4} {strat:18}"
              f"{'   << over budget' if bad else ''}")

    print(f"\n{len(rows)} configurations, {len(over)} over budget")
    if over:
        print("\nThese would be recorded as errors and leave holes in the curves. The one to")
        print("check first is D=1, which every parallel-efficiency figure divides by:")
        for gb, _, mode, d, n, dev, how, strat in over:
            if dev == 1:
                print(f"  MISSING BASELINE  {mode} d={d} N={n} how={how} {strat}"
                      f"  needs {gb:.0f} GB")
        return 1
    print("every configuration fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
