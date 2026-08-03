"""Is the trajectory guard measuring the library, or measuring GPU nondeterminism?

`lrdiag` established that at `d=256, N=64`, `lowrank_r1`, strategy A, on 2x T4:

  - the raw fitnesses are bitwise equal at D=1 and D=2,
  - there are no ties and no member changes rank,
  - **one fresh generation produces bitwise identical parameters at both device counts**,

and yet `run.py` reports `6.32e-03` for that same cell on that same node. So the divergence
is introduced by something the driver does that a single generation does not, which moves
the suspicion off the perturbation strategy and onto the harness.

Two candidates, tested here in the order that settles the most:

1. **The result is not reproducible at all.** `docs/06`'s G1 cell sets
   `--xla_gpu_deterministic_ops=true`; the phase 2 kernels never have. Some XLA:GPU
   reductions and autotuned GEMM selections vary between compilations of the same HLO. If
   two identical runs at the *same* device count disagree, the guard has been comparing
   noise to noise and "A is bitwise identical across D" is not a testable claim on a GPU
   without that flag.

2. **The warm-up changes the answer.** `run.py` compiles `generation` on the first warm-up
   call and reuses that executable for the fingerprint. The fingerprint's
   `with jax.default_matmul_precision("highest")` therefore cannot affect it, and whatever
   autotuning happened during warm-up is baked in. Step 2 fingerprints the same compiled
   function before and after warm-up.

Step 3 repeats both under `--xla_gpu_deterministic_ops=true`. If the failure disappears,
the sweep needs that flag and the guard's standard survives; if it does not, the standard
has to be restated before a paid node is booked.
"""

import os
import subprocess
import sys

SHA = "74fa602"  # same commit lrdiag ran, so the numbers are directly comparable
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

base = {**os.environ, "PYTHONPATH": "src", "JAX_PLATFORMS": "cuda"}
run([sys.executable, "-c",
     "import jax; d=jax.devices(); print(jax.__version__, len(d), d[0].device_kind);"
     " assert len(d)==2, f'{len(d)} devices, want 2'"], env=base)

PROBE = r'''
import os
import sys
import numpy as np
import jax

sys.path.insert(0, "experiments/phase2")
from shardes import sharding
from shardes.core import ShardedES
from shardes.strategies.registry import STRATEGIES
from shardes.problems import transformer_block
import run as R

SEED, SIGMA, LR = 0, 0.01, 0.05
D_MODEL, N, BATCH, SEQ = 256, 64, 8, 32
STRATEGY, HOW = "lowrank_r1", "A"

print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '(unset)')}")


def flat(params):
    return np.concatenate([np.asarray(x, np.float64).ravel()
                           for x in jax.tree.leaves(params)])


def rel(a, b):
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d else float(np.linalg.norm(a - b))


def setup(devices):
    mesh = sharding.make_mesh(devices)
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=D_MODEL)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=D_MODEL, batch=BATCH, seq=SEQ)
    es = ShardedES(STRATEGIES[STRATEGY].build(), n=N, sigma=SIGMA, lr=LR,
                   mesh=mesh, how=HOW)
    return es, params, batch


def fingerprint_after(devices, warmups):
    """One generation from a fresh state, using an executable warmed `warmups` times.

    This is exactly run.py's shape: the same compiled `generation` serves the timing loop
    and then the guard. warmups=0 is the single-generation case lrdiag measured as bitwise
    identical across D.
    """
    es, params, batch = setup(devices)

    @jax.jit
    def generation(s):
        pert, s2 = es.ask(s)
        fit = es.apply(transformer_block.loss, s2, pert)(batch)
        return es.tell(s2, pert, fit)

    with jax.default_matmul_precision("highest"):
        s = es.init(jax.random.key(SEED), params)
        for _ in range(warmups):
            s = generation(s)
        jax.block_until_ready(s)
        fresh = generation(es.init(jax.random.key(SEED), params))
        jax.block_until_ready(fresh)
    return flat(fresh.params)


print("\n== step 1: is one device count even reproducible run to run? ==")
for devices in (1, 2):
    a = fingerprint_after(devices, 0)
    b = fingerprint_after(devices, 0)
    same = np.array_equal(a, b)
    print(f"  D={devices}: two identical fresh runs agree bitwise: {same}"
          f"   L2 rel {rel(a, b):.3e}"
          f"{'' if same else '   <-- NONDETERMINISTIC, the guard compares noise'}")

print("\n== step 2: does the warm-up change the fingerprint? ==")
by = {}
for devices in (1, 2):
    for warmups in (0, 8):
        by[(devices, warmups)] = fingerprint_after(devices, warmups)
for warmups in (0, 8):
    d = rel(by[(2, warmups)], by[(1, warmups)])
    flag = "   <-- the guard failure" if d > 1e-5 else ""
    print(f"  warmups={warmups}: D=1 vs D=2  L2 rel {d:.3e}"
          f"   exact={np.array_equal(by[(2, warmups)], by[(1, warmups)])}{flag}")
for devices in (1, 2):
    d = rel(by[(devices, 8)], by[(devices, 0)])
    print(f"  D={devices}: warmed vs cold, same device count  L2 rel {d:.3e}")

print("\n== step 3: the driver's own measure(), twice, same device count ==")
cfg = {"batch": BATCH, "seq": SEQ, "warmup": 3, "repeats": 5,
       "matmul_precision": "highest"}
for devices in (1, 2):
    out = []
    for _ in range(2):
        c = R.Config(mode="strong", devices=devices, d_model=D_MODEL, population=N,
                     strategy=STRATEGY, how=HOW)
        out.append(R.measure(c, cfg)["trajectory"])
    same = out[0]["digest"] == out[1]["digest"]
    p0, p1 = np.array(out[0]["probe"]), np.array(out[1]["probe"])
    print(f"  D={devices}: measure() twice agrees: {same}   probe L2 rel {rel(p1, p0):.3e}"
          f"{'' if same else '   <-- measure() is not reproducible'}")
    by[("measure", devices)] = np.array(out[0]["probe"])

d = rel(by[("measure", 2)], by[("measure", 1)])
print(f"\n  measure(): D=1 vs D=2 probe L2 rel {d:.3e}"
      f"{'   <-- reproduces the 6.32e-03 failure' if d > 1e-5 else '   <-- does NOT reproduce'}")
'''

with open("nondet_inner.py", "w") as fh:
    fh.write(PROBE)

print("\n" + "=" * 72)
print("PASS 1: as the phase 2 sweep actually runs, no determinism flag")
print("=" * 72)
run([sys.executable, "nondet_inner.py"], env=base, check=False)

print("\n" + "=" * 72)
print("PASS 2: with --xla_gpu_deterministic_ops=true, as docs/06's G1 cell sets")
print("=" * 72)
det = {**base, "XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}
run([sys.executable, "nondet_inner.py"], env=det, check=False)

print("\nReading it:")
print("  step 1 nondeterministic          -> the guard has been comparing noise; the")
print("                                      'A is bitwise identical' standard needs the")
print("                                      determinism flag to be testable at all.")
print("  step 1 clean, step 2 warmup-only -> the harness bakes autotuning from the timing")
print("                                      loop into the guard; fingerprint on its own")
print("                                      compiled function.")
print("  pass 2 clean, pass 1 not         -> the sweep needs the flag; the standard holds.")
print("\nnondet done")
