#!/usr/bin/env python
"""TB3, the ablation table, assembled from committed results. No new measurement.

    python tb3.py            # markdown to stdout

Every number a reviewer might ask "did you check X" about, computed from the
directory that already answers it, so the table is reproducible from a clean
checkout (CLAUDE.md ground rule 2). Rows whose source directory is absent print
PENDING with the session that fills them, rather than silently vanishing: a
table missing a row reads as "didn't check" when the truth is "not yet run".

Ratios are geometric means over the cells where BOTH sides were measured, since
cost ratios multiply and the grid spans two orders of magnitude in wall time.
The σ and estimator-quality axes live in phase 0 (E1, F5) and are cited, not
recomputed: this script owns the systems rows.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
COUNTDOWN = HERE.parent / "countdown" / "results" / "e13-a100-2026-08-22-clean"

GPU = HERE / "results-cost"
TPU = HERE / "results-cost-tpu-v5e8"
TPU_HIGHEST = HERE / "results-cost-tpu-v5e8-highest"


def cells(d: pathlib.Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        c = r["config"]
        key = (c["d_model"], c["population"], c["strategy"], c["dtype"])
        out[key] = None if r.get("undersized") else r["seconds_median"]
    return out


def geomean(ratios: list[float]) -> float:
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


def ratio_row(cs: dict, num_sel, den_sel) -> str:
    """Geometric-mean ratio over cells where both selector variants measured."""
    rs = []
    for k, t in cs.items():
        nk = num_sel(k)
        if nk is None or t is None:
            continue
        base = cs.get(den_sel(k) if den_sel else k)
        num = cs.get(nk)
        if num and base:
            rs.append(num / base)
    return f"{geomean(rs):.2f}x (n={len(rs)})" if rs else "no overlapping cells"


def rank_cost(cs: dict, platform: str) -> list[str]:
    rows = []
    for r in (1, 4, 16):
        val = ratio_row(
            cs,
            lambda k, r=r: (k[0], k[1], f"mirrored_lr{r}", k[2 + 1])
            if k[2] == "iid_gaussian" and k[3] == "bfloat16" else None,
            None,
        )
        rows.append(f"| cost, rank {r} vs dense | {platform}, bf16 | {val} |")
    return rows


def dtype_cost(cs: dict, platform: str) -> list[str]:
    rows = []
    for s in ("iid_gaussian", "seed_regenerated", "mirrored_lr1"):
        val = ratio_row(
            cs,
            lambda k, s=s: (k[0], k[1], s, "bfloat16")
            if k[2] == s and k[3] == "float32" else None,
            None,
        )
        rows.append(f"| cost, bf16 vs f32 | {platform}, {s} | {val} |")
    return rows


def precision_cost() -> list[str]:
    if not TPU_HIGHEST.exists() or not any(TPU_HIGHEST.glob("*.json")):
        return ["| cost, `highest` vs `default` | TPU v5e | "
                "PENDING: T4 session, cost-precision-tpu.yaml |"]
    hi, lo = cells(TPU_HIGHEST), cells(TPU)
    rows = []
    for s in sorted({k[2] for k in hi}):
        rs = [hi[k] / lo[k] for k in hi
              if k[2] == s and hi.get(k) and lo.get(k)]
        if rs:
            rows.append(f"| cost, `highest` vs `default` | TPU v5e, {s} | "
                        f"{geomean(rs):.2f}x (n={len(rs)}) |")
    return rows


def task_quality() -> list[str]:
    if not COUNTDOWN.exists():
        return ["| task quality vs rank | Countdown | PENDING: e13 results missing |"]
    rows = []
    for stem, label in (("es-mirrored-seed", "full rank"), ("es-mirrored-lr1", "rank 1"),
                        ("es-mirrored-lr4", "rank 4"), ("es-mirrored-lr16", "rank 16")):
        finals = []
        for s in (0, 1, 2):
            lines = (COUNTDOWN / f"{stem}-s{s}-eval.jsonl").read_text().splitlines()
            finals.append(json.loads(lines[-1])["eval_reward"])
        # Mean, matching F7 and the campaign README; the statistic is named
        # in the caption because an unnamed summary is unreadable.
        rows.append(f"| held-out reward, {label} | Countdown, 3 seeds | "
                    f"{statistics.mean(finals):.3f} "
                    f"[{min(finals):.3f}, {max(finals):.3f}] |")
    return rows


def main() -> int:
    header = ["| ablation | where | result |", "|---|---|---|"]
    # Systems rows: timing ratios only. The validation rows (quality,
    # guards, invariance) answer a different question ("was it checked")
    # and go to their own table in the appendix.
    systems = []
    for d, platform in ((GPU, "A100"), (TPU, "TPU v5e")):
        if d.exists():
            systems += rank_cost(cells(d), platform)
    for d, platform in ((GPU, "A100"), (TPU, "TPU v5e")):
        if d.exists():
            systems += dtype_cost(cells(d), platform)
    systems += precision_cost()
    validation = task_quality() + [
        "| update--gradient alignment vs rank, σ, N | phase 0 | E1 / F5: "
        "`experiments/phase0/` (measured, committed) |",
        "| sub-f32 fitness | guard | refused by `tell`; measured collapse "
        "256 losses -> 2 in bf16 (docs/proposal-bf16-policy.md) |",
        "| device-count invariance | tests | `tests/`: D=1 vs D=8 within "
        "rtol 1e-5 bf16 / 1e-12 f32; C6d multi-GPU run |",
    ]
    print("\n".join(header + systems))
    print()
    print("\n".join(header + validation))

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    if ap.parse_args().latex:
        # The markdown rows are the single source; convert rather than rebuild,
        # so the two outputs cannot drift.
        def esc(s):
            return (s.replace("\\", "").replace("&", "\\&").replace("_", "\\_")
                    .replace("σ", "$\\sigma$").replace("->", "$\\to$")
                    .replace("`", "").replace("%", "\\%"))

        def emit(rows, caption, label, fname):
            body = []
            for ln in rows:
                cellsx = [c.strip() for c in ln.strip("|").split("|")]
                body.append(" & ".join(esc(c) for c in cellsx) + " \\\\")
            tex = "\n".join([
                "% generated by experiments/phase2/tb3.py; do not edit by hand",
                "\\begin{table*}", "\\centering", "\\small",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{p{0.30\\linewidth}p{0.22\\linewidth}p{0.40\\linewidth}}",
                "\\toprule", "ablation & where & result \\\\", "\\midrule",
                *body,
                "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
            dest = HERE.parent.parent / "paper" / "generated" / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(tex)
            print(f"wrote {dest}")

        emit(systems,
             "Systems ablations, assembled from committed results; ratios "
             "are geometric means over the grid cells where both sides were "
             "measured (n = cell count).",
             "tab:tb3", "tb3.tex")
        emit(validation,
             "Validation checklist, assembled from committed results: the "
             "quality, guard, and invariance checks behind the systems "
             "claims; held-out rewards are means over seeds with [min, max].",
             "tab:tb3b", "tb3b.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
