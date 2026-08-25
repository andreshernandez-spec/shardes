#!/usr/bin/env python
"""The contraction crossover as a time model, calibrated per fabric.

    python timemodel.py                       # the table, both platforms
    python timemodel.py --latex               # also writes ../../paper/generated/tb7.tex
    python timemodel.py --fabric 120e-6,12.5e9 --label "25 GbE"   # predict another fabric

Section 4 of the paper commits a byte count before the measurement and Section 6 reports
end-to-end seconds. Nothing converts one into the other, so the sentence "B wins whenever
the contraction it avoids costs more than those bytes" is an assertion: bytes reach time
through an achieved bandwidth that is itself a function of message size, and the same byte
counts give a different crossover on the A100 and the v5e. This is the missing term.

The model has one line and no free parameters once the two harnesses have run:

    t_A - t_B = C(N, d) (D - 1) / D  +  ag(4N)  -  ar(4P)

  C(N, d)   the replicated contraction, measured by `contraction_isolation.py`
  ar, ag    the two collectives at their real payloads, measured by `allreduce_ladder.py`
            as an in-program step cost, alpha + bytes / beta between its grid points
  P = 6 d^2 for the block (six square matrices, `problems/transformer_block.py`)

`--calibrate` runs it backwards instead, solving each measured cell for the C that would
close it. That mode needs no contraction records and is what the residual table reports:
where the solved C disagrees with the measured one, the difference is cost the two isolated
harnesses do not see, and saying so is the point of the exercise.

WHAT THIS IS AND IS NOT. Section 4's byte model was committed before E4 and E7, and this
one was not: it is calibrated on the single-host grids it is checked against, so it is an
explanation, not a prediction. `--fabric` is where it becomes a prediction, and
`docs/11-e18b-preregistration.md` is that prediction written down before the rental.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: Sweep directories per platform, and the ladder record that calibrates each fabric.
PLATFORMS = {
    "8x A100-SXM4-80GB (NVLink)": {
        "sweeps": ("results-consistent", "results-qiu"),
        "ladder": "results-ladder/ladder-nvidia-a100-sxm4-80gb-D8.json",
        "kind": "nvidia-a100-sxm4-80gb",
    },
    "TPU v5e-8 (ICI)": {
        "sweeps": ("results-tpu-v5e8",),
        "ladder": "results-ladder/ladder-tpu-v5-lite-D8.json",
        "kind": "tpu-v5-lite",
    },
}

DEVICES = 8
BYTES_PER_F32 = 4


def params_bytes(d_model: int) -> int:
    """Six square matrices, float32. 6 MiB at d=512, 96 MiB at d=2048."""
    return 6 * d_model * d_model * BYTES_PER_F32


class Fabric:
    """alpha and beta for the two collectives, from the ladder's in-program step costs.

    Piecewise-linear between measured points rather than a single alpha-beta fit: the
    A100's step cost is flat from 8 B to 1 MiB and then linear, which one line through the
    endpoints gets wrong by 2x in the middle. Both payloads the placements actually issue
    (4P at d=512 and d=2048, 4N at N=256 and 1024) are measured points, so the interpolation
    only ever fills the gaps a hypothetical fabric asks for.
    """

    def __init__(self, allreduce: dict, allgather_n: dict, label: str):
        self.label = label
        self.ar = sorted((int(k), v) for k, v in allreduce.items())
        self.ag = sorted((4 * int(k), v) for k, v in allgather_n.items())

    @staticmethod
    def _interp(points, nbytes):
        if nbytes <= points[0][0]:
            return points[0][1]
        for (b0, t0), (b1, t1) in zip(points, points[1:]):
            if nbytes <= b1:
                return t0 + (t1 - t0) * (nbytes - b0) / (b1 - b0)
        (b0, t0), (b1, t1) = points[-2], points[-1]
        return t1 + (t1 - t0) * (nbytes - b1) / (b1 - b0)

    def allreduce(self, nbytes: int) -> float:
        return self._interp(self.ar, nbytes)

    def allgather(self, nbytes: int) -> float:
        return self._interp(self.ag, nbytes)

    @classmethod
    def from_ladder(cls, path: pathlib.Path, label: str) -> "Fabric":
        rec = json.loads(path.read_text())
        ar = {k: v["step_seconds"] for k, v in rec["allreduce"].items()}
        ag = {k: v["step_seconds"] for k, v in rec["allgather"].items()}
        return cls(ar, ag, label)

    @classmethod
    def synthetic(cls, alpha: float, beta: float, label: str) -> "Fabric":
        """A hypothetical fabric from one alpha (s) and one beta (bytes/s). Both collectives
        get the same pair: for a prediction about a slow link the model-sized all-reduce is
        the only term that matters, and the 4N gather is latency either way."""
        sizes = (8, 2**10, 2**20, 2**26, 100 * 2**20)
        ar = {str(b): {"step_seconds": alpha + b / beta} for b in sizes}
        ag = {str(n): {"step_seconds": alpha + 4 * n / beta} for n in (256, 1024, 2**18)}
        return cls({k: v["step_seconds"] for k, v in ar.items()},
                   {k: v["step_seconds"] for k, v in ag.items()}, label)


def load_sweep(dirs, devices=DEVICES) -> dict:
    """(strategy, d, N) -> {"A": seconds, "B": seconds} at D=devices, strong scaling."""
    cells = collections.defaultdict(dict)
    for name in dirs:
        for path in (HERE / name).glob("*.json"):
            rec = json.loads(path.read_text())
            cfg = rec.get("config", {})
            if cfg.get("mode") != "strong" or cfg.get("devices") != devices:
                continue
            if "seconds_median" not in rec:
                continue
            key = (cfg["strategy"], cfg["d_model"], cfg["population"])
            cells[key][cfg["how"]] = rec["seconds_median"]
    return {k: v for k, v in cells.items() if "A" in v and "B" in v}


def load_contraction(kind: str, devices=DEVICES) -> dict:
    """(strategy, d, N) -> the isolation record, if `contraction_isolation.py` has run."""
    out = {}
    for path in (HERE / "results-contraction").glob(f"*__{kind}__D{devices}.json"):
        rec = json.loads(path.read_text())
        if "failed" in rec:
            continue
        cfg = rec["config"]
        out[(cfg["strategy"], cfg["d_model"], cfg["population"])] = rec
    return out


def rows(platform: str, spec: dict, devices=DEVICES, fabric: Fabric | None = None) -> list:
    sweep = load_sweep(spec["sweeps"], devices)
    contr = load_contraction(spec["kind"], devices)
    fab = fabric or Fabric.from_ladder(HERE / spec["ladder"], platform)
    out = []
    for (strategy, d_model, n), t in sorted(sweep.items()):
        ar = fab.allreduce(params_bytes(d_model))
        ag = fab.allgather(BYTES_PER_F32 * n)
        delta = t["A"] - t["B"]
        # Backwards: the contraction that would close this cell.
        c_solved = (delta - ag + ar) * devices / (devices - 1)
        rec = contr.get((strategy, d_model, n))
        c_meas = rec["contraction_seconds"] if rec else None
        # With the isolation records the two modelled terms become measured ones: the
        # saving is C - C_local, not C (D-1)/D, because the contraction does not shard
        # at 1/D; and the collective is the psum this program actually issues, not the
        # ladder's psum in an empty one.
        if rec is not None:
            saving = rec["contraction_seconds"] - rec["contraction_local_seconds"]
            pred = saving + ag - rec["allreduce_insitu_seconds"]
        else:
            pred = None
        out.append({
            "strategy": strategy, "d_model": d_model, "population": n,
            "t_A": t["A"], "t_B": t["B"], "delta_measured": delta,
            "allreduce_seconds": ar, "allgather_seconds": ag,
            "params_bytes": params_bytes(d_model),
            "contraction_solved": c_solved,
            "contraction_measured": c_meas,
            "contraction_local_measured": rec["contraction_local_seconds"] if rec else None,
            "shard_ratio": rec["shard_ratio"] if rec else None,
            "allreduce_insitu": rec["allreduce_insitu_seconds"] if rec else None,
            "allreduce_insitu_over_ladder": (rec["allreduce_insitu_seconds"] / ar) if rec else None,
            "delta_predicted": pred,
            "sign_agrees": None if pred is None else ((pred > 0) == (delta > 0)),
        })
    return out, fab


def crossover_population(rows_, d_model: int, devices=DEVICES) -> dict:
    """N* where the model flips, per strategy, from a linear fit of C in N at this d.

    C is taken as proportional to N (one perturbation contracted per member), fitted through
    the origin on whatever cells exist at this d, so two cells give a slope and one gives a
    ray. The flip is at C(N) (D-1)/D = ar - ag, i.e. at the population where the contraction
    B avoids finally pays for the transfer it adds.
    """
    out = {}
    by_strategy = collections.defaultdict(list)
    for r in rows_:
        if r["d_model"] != d_model:
            continue
        c = r["contraction_measured"] if r["contraction_measured"] is not None \
            else r["contraction_solved"]
        by_strategy[r["strategy"]].append((r["population"], c, r))
    for strategy, pts in by_strategy.items():
        num = sum(n * c for n, c, _ in pts)
        den = sum(n * n for n, _, _ in pts)
        if den == 0:
            continue
        per_member = num / den  # least squares through the origin
        r0 = pts[0][2]
        need = r0["allreduce_seconds"] - r0["allgather_seconds"]
        slope = per_member * (devices - 1) / devices
        out[strategy] = {
            "contraction_per_member_seconds": per_member,
            "n_star": (need / slope) if slope > 0 else float("inf"),
        }
    return out


def render(platform, rows_, fab, devices=DEVICES) -> str:
    have_model = any(r["delta_predicted"] is not None for r in rows_)
    lines = [f"## {platform}, D={devices}", ""]
    head = ("| strategy | d | N | t_A ms | t_B ms | measured A-B ms | all-reduce ms "
            "| C solved ms |")
    rule = "|---|---|---|---|---|---|---|---|"
    if have_model:
        head += " C measured ms | predicted A-B ms | sign |"
        rule += "---|---|---|"
    lines += [head, rule]
    for r in rows_:
        row = ("| %s | %d | %d | %.2f | %.2f | %+.2f | %.3f | %.2f |"
               % (r["strategy"], r["d_model"], r["population"], r["t_A"] * 1e3,
                  r["t_B"] * 1e3, r["delta_measured"] * 1e3,
                  r["allreduce_seconds"] * 1e3, r["contraction_solved"] * 1e3))
        if have_model:
            if r["delta_predicted"] is None:
                row += " | | | |"
            else:
                row += (" %.2f | %+.2f | %s |"
                        % (r["contraction_measured"] * 1e3, r["delta_predicted"] * 1e3,
                           "ok" if r["sign_agrees"] else "MISS"))
        lines.append(row)
    lines.append("")

    # A contraction cannot cost less than nothing. Where the solved C goes negative, B is
    # slower than the local contraction plus the ladder's all-reduce, so the two isolated
    # harnesses do not account for everything B pays. That deficit is the model's open term
    # and `contraction_isolation.py`'s `allreduce_insitu` is what closes or confirms it.
    deficit = [r for r in rows_ if r["contraction_solved"] < 0]
    if deficit:
        worst = min(deficit, key=lambda r: r["contraction_solved"])
        lines.append(
            "unexplained: %d of %d cells solve to a negative contraction; worst "
            "%s d=%d N=%d at %.2f ms, where B costs %.2f ms more than the ladder's "
            "all-reduce accounts for"
            % (len(deficit), len(rows_), worst["strategy"], worst["d_model"],
               worst["population"], worst["contraction_solved"] * 1e3,
               -worst["contraction_solved"] * (devices - 1) / devices * 1e3))
    resid = [r for r in rows_ if r["delta_predicted"] is not None]
    if resid:
        miss = [r for r in resid if not r["sign_agrees"]]
        worst = max(resid, key=lambda r: abs(r["delta_measured"] - r["delta_predicted"]))
        lines.append(
            "forward model: %d of %d cells predicted with the right sign; largest residual "
            "%.2f ms (%s d=%d N=%d)"
            % (len(resid) - len(miss), len(resid),
               (worst["delta_measured"] - worst["delta_predicted"]) * 1e3,
               worst["strategy"], worst["d_model"], worst["population"]))
    lines.append("")
    for d_model in sorted({r["d_model"] for r in rows_}):
        star = crossover_population(rows_, d_model, devices)
        for strategy, v in sorted(star.items()):
            lines.append("N* (d=%d, %s): %s members"
                         % (d_model, strategy,
                            "never" if v["n_star"] == float("inf")
                            else f"{v['n_star']:.0f}"))
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=DEVICES)
    ap.add_argument("--fabric", help="alpha_seconds,beta_bytes_per_second for a hypothetical link")
    ap.add_argument("--label", default="hypothetical fabric")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args(argv)

    fabric = None
    if args.fabric:
        alpha, beta = (float(x) for x in args.fabric.split(","))
        fabric = Fabric.synthetic(alpha, beta, args.label)

    dump, text = {}, []
    for platform, spec in PLATFORMS.items():
        if not (HERE / spec["ladder"]).exists():
            print(f"skip {platform}: no ladder record", file=sys.stderr)
            continue
        rs, fab = rows(platform, spec, args.devices, fabric)
        if not rs:
            continue
        label = platform if fabric is None else f"{platform} sweep on {fabric.label}"
        text.append(render(label, rs, fab, args.devices))
        dump[label] = rs
    out = "\n".join(text)
    print(out)
    if args.json:
        args.json.write_text(json.dumps(dump, indent=1))
    if args.latex:
        print("latex output needs the measured contraction column; run "
              "contraction_isolation.py first", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
