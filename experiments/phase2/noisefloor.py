#!/usr/bin/env python
"""Can a configuration tell its own members apart, or is it ranking rounding noise?

    python noisefloor.py --config sweep.yaml        # non-zero exit if any config is too close

`check.py` asks whether two device counts computed the same thing. This asks a different and
prior question: whether the thing being computed is well enough separated to be worth
comparing at all. A sweep can pass the trajectory guard and still be measuring noise.

**Why this exists.** The phase 2 sweep failed device-count invariance on `lowrank_r1` at
`d=256, N=64`, by 6.32e-03 under contraction strategy A, which is supposed to be bitwise
identical. The chain turned out to be:

1. XLA:GPU picks reduction algorithms per shape unless `--xla_gpu_deterministic_ops=true`
   is set, so scoring 64 members in one batch and 32 in each of two batches can use
   different summation orders. That perturbs each score by roughly one ulp.
2. At that configuration the two closest members were **2.00 ulp apart**.
3. One ulp of noise is therefore enough to swap them.
4. `centered_ranks` sorts, so a swap moves a whole rank step. Measured amplification from a
   single swap: **8.93e-03 out of a 3.37e-08 input change, a factor of 2.65e5.** That
   matches the observed failure to three significant figures.

The determinism flag removes step 1 and `run.py` now requires it. It does nothing about
steps 2 to 4. A different GPU, a newer XLA or cuBLAS, or TF32 instead of fp32 all perturb
scores by a comparable amount, and the amplification is untouched. So the flag makes the
sweep *reproducible*, not *well conditioned*, and those are different claims.

**The criterion.** A configuration whose closest two members sit within a few ulp of each
other cannot distinguish them: their relative order is decided by rounding, not by fitness.
The update is still a valid ES update (two statistically indistinguishable members swapping
places is harmless for convergence) but it is not a reproducible one, and a benchmark that
quotes device-count invariance has to say which it means.

**The margin is empirical, not derived.** Measured: a flip at 2.00 ulp; no flips at 21, 47
or 63 ulp. 16 is a fence inside that band. A tighter bound would have to model the summation
error of the whole forward pass, which depends on the reduction tree XLA picks and is
exactly the thing that is not fixed across shapes. Treat the printed numbers as the result
and the pass/fail as a convenience.

Device count is not swept here: score separation is a property of the model, the population
and the strategy, not of how the work is divided. `how` is not swept either, for the same
reason. That makes this 16 probes for the 256-configuration sweep rather than 256.

**Memory.** The probe holds the whole population on one device, so a shape that needs
sharding to fit in the sweep needs a big single device here. `d=2048, N=256` with
`iid_gaussian` wants about 50 GB. Run the large end where the sweep runs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax  # noqa: E402

import run as R  # noqa: E402  SEED, SIGMA, LR and STRATEGIES, so they cannot drift

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402

#: Below this many ulp between the closest pair, the ordering is rounding rather than
#: fitness. See the module docstring: measured flip at 2 ulp, clean at 21 and above.
MARGIN_ULP = 16.0


def scores(d_model: int, population: int, strategy: str, seed: int,
           batch: int = 8, seq: int = 32) -> np.ndarray:
    """One generation's fitness, set up exactly as `run.measure` does.

    Single device and strategy A deliberately: this measures the separation of the scores,
    which is a property of the problem and the perturbation scheme. Splitting the population
    across devices does not change what the members are.
    """
    key = jax.random.key(seed)
    params = transformer_block.init(key, d_model=d_model)
    data = transformer_block.make_batch(
        jax.random.fold_in(key, 1), d_model=d_model, batch=batch, seq=seq
    )
    es = ShardedES(R.STRATEGIES[strategy](), n=population, sigma=R.SIGMA, lr=R.LR,
                   mesh=sharding.make_mesh(1), how="A")
    state = es.init(key, params)

    @jax.jit
    def once(state):
        pert, s = es.ask(state)
        return es.apply(transformer_block.loss, s, pert)(data)

    with jax.default_matmul_precision("highest"):
        return np.asarray(jax.device_get(once(state)), np.float32)


def separation(f: np.ndarray, margin: float = MARGIN_ULP) -> dict:
    """Gap statistics in units of the last place, which is the only scale that matters here.

    `np.spacing` of the typical score, not of each score: the reduction error is set by the
    magnitude of the accumulation, and all the scores here are the same magnitude. Using a
    per-score ulp would report a different noise floor for each member and none of them
    would be the one that actually applies.
    """
    ulp = float(np.spacing(np.float32(float(np.mean(np.abs(f))))))
    gaps = np.diff(np.sort(f.astype(np.float64)))
    return {
        "loss": float(np.mean(np.abs(f))),
        "ulp": ulp,
        "min_gap": float(gaps.min()),
        "min_gap_ulp": float(gaps.min()) / ulp,
        "ties": int((gaps == 0.0).sum()),
        "below_margin": int((gaps < margin * ulp).sum()),
        "pairs": int(gaps.size),
    }


def shapes(cfg: dict) -> list[tuple[int, int, str]]:
    """Distinct (d_model, population, strategy). Both modes contribute populations."""
    from feasible import populations  # noqa: PLC0415

    out = set()
    for d_model in cfg["d_model"]:
        for mode in cfg["modes"]:
            for devices in cfg["devices"]:
                for population in populations(cfg, mode, d_model, devices):
                    for strategy in cfg["strategies"]:
                        out.add((d_model, population, strategy))
    return sorted(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=pathlib.Path, default=HERE / "sweep.yaml")
    ap.add_argument("--margin", type=float, default=MARGIN_ULP,
                    help="minimum acceptable gap between the closest two members, in ulp")
    ap.add_argument("--seeds", type=int, default=3,
                    help="draws to sample; the worst is reported, since one draw is a sample "
                         "of the configuration rather than a property of it")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    batch, seq = int(cfg.get("batch", 8)), int(cfg.get("seq", 32))

    print(f"{args.config.name}, margin {args.margin:g} ulp, worst of {args.seeds} seeds "
          f"(sweep runs seed {R.SEED})\n")
    print(f"{'d':>6}{'N':>6}  {'strategy':18}{'loss':>10}{'min gap':>11}"
          f"{'ulp':>9}{'ties':>6}{'close':>7}")

    rows, failures = [], []
    for d_model, population, strategy in shapes(cfg):
        worst = None
        for seed in range(R.SEED, R.SEED + args.seeds):
            s = separation(scores(d_model, population, strategy, seed, batch, seq),
                           args.margin)
            if worst is None or s["min_gap_ulp"] < worst["min_gap_ulp"]:
                worst = s
        rows.append((d_model, population, strategy, worst))
        bad = worst["min_gap_ulp"] < args.margin
        if bad:
            failures.append((d_model, population, strategy, worst))
        print(f"{d_model:>6}{population:>6}  {strategy:18}{worst['loss']:>10.4f}"
              f"{worst['min_gap']:>11.2e}{worst['min_gap_ulp']:>9.2f}"
              f"{worst['ties']:>6}{worst['below_margin']:>7}"
              f"{'   TOO CLOSE' if bad else ''}")

    print()
    if failures:
        print(f"{len(failures)} of {len(rows)} configurations cannot separate their closest "
              f"two members by {args.margin:g} ulp.\n")
        print("Their ranking of those members is decided by rounding, so the update is not")
        print("reproducible across anything that perturbs the scores: a different GPU, a")
        print("newer XLA, TF32 instead of fp32. One swap is worth ~1e-2 in the update.")
        print()
        print("An exact tie is not safer than a near miss. `centered_ranks` gives tied members")
        print("*different* weights, chosen by the sort's tie-break, so a pair that ties on one")
        print("backend and differs by an ulp on another can order either way.")
        print()
        print("Remedies, in the order they were measured rather than the order they occur:")
        print("  - Raising sigma works when N is small and does NOT work when N is large.")
        print("    lowrank_r1 at d=256 N=64 goes 0 -> 56 ulp from sigma 0.01 -> 0.03, but")
        print("    iid_gaussian at d=256 N=256 goes 1 -> 8 -> 6 -> 2 ulp across 0.01 -> 0.3,")
        print("    because the loss magnitude grows with sigma and takes the ulp with it.")
        print("  - The binding constraint is N against float32 resolution. Separating N")
        print("    members by m ulp needs a relative loss spread of about N*m*eps, which is")
        print("    5e-4 at N=256, m=16. Large populations are the hard case, and this sweep")
        print("    goes to N=1024.")
        print("  - Midrank shaping (tied members share the average rank) would make exact")
        print("    ties harmless, though not near-ties. That is estimator math, so it is a")
        print("    proposal rather than something this script assumes.")
        print("  - Or keep the configuration, pin the arithmetic, and state in the")
        print("    limitations that invariance holds under fixed arithmetic only.")
        return 1
    print(f"every configuration separates its closest two members by at least "
          f"{args.margin:g} ulp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
