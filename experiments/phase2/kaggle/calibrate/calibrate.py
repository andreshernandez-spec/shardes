"""How long is the Phase 2 sweep, and does the memory model hold? Answered for free.

Booking an 8x A100 node without this is how a $50 run becomes a $200 one. The GPU tier
queues in seconds, unlike T1's 2.5-4 h, so this is the cheap way to size the paid session.

Two questions, and the second matters more:

1. **Wall clock per configuration at real shapes.** The rehearsal ran d=32..64 in
   milliseconds, which says nothing about d=512 or 2048 where compile time and the matmuls
   both grow. Extrapolating from toy shapes is how a 4-hour estimate becomes 12.

2. **Does measured peak memory match `feasible.py`?** That model is analytic: strategy A
   stores `N*|params|` on every device, B stores `N/D*|params|`. The whole sweep was resized
   on the strength of it, so it is worth one measurement against
   `compiled.memory_analysis()` rather than trusting arithmetic that has never been checked.

Runs on 2x T4, 16 GB each, so only the small end of the sweep fits. That is fine: the point
is a cost *model*, fitted on what fits and extrapolated with its shape stated, not a
simulation of the real run.
"""

import json
import os
import subprocess
import sys

SHA = "b75c0c3"  # SET THIS to the commit under test.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]>=0.11"])
run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

# `--xla_gpu_deterministic_ops=true` is not optional and run.py now refuses without it.
# Without it XLA selects reduction algorithms per shape, so D=1 and D=2 run different
# arithmetic; the first calibration run failed the guard at 6.32e-03 on a configuration that
# is exactly zero with the flag set. It may cost throughput, and that is the trade: these
# timings are for deterministic reductions, which is the only configuration whose
# correctness can be checked. The previous run's numbers are the comparison.
env = {**os.environ, "PYTHONPATH": "src", "JAX_PLATFORMS": "cuda",
       "XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}

run([sys.executable, "-c",
     "import jax; d=jax.devices(); print(jax.__version__, len(d), d[0].device_kind);"
     " assert len(d)==2, f'{len(d)} devices, want 2'"], env=env)

# A config the T4s can actually hold, spanning d and N so the cost model has a slope in both.
# `feasible.py` says A at d=512,N=1024 needs 6.4 GB a device, which fits 16 GB; d=2048 does
# not, so it is deliberately absent and the extrapolation has to carry that gap.
cal = {
    "modes": ["strong"],
    "devices": [1, 2],
    "d_model": [256, 512],
    "population": {256: [64, 256], 512: [64, 256]},
    "population_per_device": {256: [32], 512: [32]},
    "strategies": ["iid_gaussian", "seed_regenerated", "mirrored_lr1", "lowrank_r1"],
    "how": ["A", "B"],
    "batch": 8, "seq": 32, "warmup": 3, "repeats": 5,
    "matmul_precision": "highest",
    "results_dir": "results-calibration",
}
with open("experiments/phase2/calibration.yaml", "w") as fh:
    json.dump(cal, fh)          # YAML is a superset of JSON, so this loads as-is

print("\n== feasibility, 16 GB per T4 ==")
run([sys.executable, "experiments/phase2/feasible.py",
     "--config", "experiments/phase2/calibration.yaml", "--hbm", "16"], env=env, check=False)

print("\n== the calibration sweep ==")
run([sys.executable, "experiments/phase2/run.py",
     "--config", "experiments/phase2/calibration.yaml", "--hbm", "16",
     "--budget", "5400"], env=env)

print("\n== guard ==")
run([sys.executable, "experiments/phase2/check.py",
     "--results", "experiments/phase2/results-calibration"], env=env, check=False)

print("\n== COST MODEL: measured time, and predicted vs measured memory ==")
run([sys.executable, "-c", '''
import glob, json, sys
sys.path.insert(0, "experiments/phase2")
from feasible import per_device_bytes
GB = 1024**3
rows = [json.load(open(f)) for f in sorted(glob.glob("experiments/phase2/results-calibration/*.json"))]
rows = [r for r in rows if "error" not in r]
print(f"{"d":>5} {"N":>5} {"D":>2} {"how":>4} {"strategy":18} {"ms":>9} {"pred GB":>8} {"meas GB":>8}")
total = 0.0
for r in sorted(rows, key=lambda r: (r["config"]["d_model"], r["config"]["population"])):
    c = r["config"]
    # Pass the strategy. Without it this defaults to iid_gaussian, the largest, and every
    # other row reads as the model over-predicting by up to 300x: seed_regenerated at
    # d=512 N=256 showed "3.19 predicted / 0.01 measured" when the model says 0.01.
    pred = per_device_bytes(c["d_model"], c["population"], c["devices"], c["how"],
                            c["strategy"]) / GB
    meas = (r.get("peak_bytes_per_device") or 0) / GB
    total += r["wall_seconds"]
    print(f"{c["d_model"]:>5} {c["population"]:>5} {c["devices"]:>2} {c["how"]:>4} "
          f"{c["strategy"]:18} {r["seconds_median"]*1e3:>9.1f} {pred:>8.2f} {meas:>8.2f}")
print(f"\\nconfigs: {len(rows)}   total wall incl. compile: {total/60:.1f} min")
print(f"mean per config: {total/max(1,len(rows)):.1f} s")
'''], env=env, check=False)
