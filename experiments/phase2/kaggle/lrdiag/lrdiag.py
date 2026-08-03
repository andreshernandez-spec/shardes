"""Why does `lowrank_r1` disagree across device counts on a GPU and nowhere else?

The calibration sweep (2x T4, one process, one backend) failed the trajectory guard on
exactly one cell: `lowrank_r1`, `d=256`, `N=64`, `6.32e-03` between `D=1` and `D=2`, under
contraction strategy A, which is supposed to be *bitwise* identical. `d=512 N=64` and
`d=256 N=256` were exact, and every other strategy was exact. On CPU-8 the same cell is
bitwise identical at every device count, so this does not reproduce off a GPU.

**The hypothesis this kernel tests.** Under A each device regenerates the whole population
and runs the same `einsum`, so the contraction cannot depend on `D`. The only sharded input
left is the fitness: it is evaluated `N/D` per device and all-gathered. A vmap over 64
members and a vmap over 32 are different shapes, and XLA:GPU may pick differently-reducing
GEMM kernels for them, so the fitnesses are permitted to differ in the last ulp. The default
shaping is `centered_ranks`, a global sort, which is *discontinuous*: one near-tie changing
order moves a whole rank step. That would produce what was measured, a direction that moves
far more than the magnitude (norm agreed to 3e-5, the probe to only 6e-3).

So: **do the raw fitnesses differ bitwise between D=1 and D=2 on a GPU, and does the rank
order change?** If yes, the strategy is not at fault and the guard's "A is bitwise
identical" standard is unattainable on GPU for any strategy whose fitnesses can near-tie.
If the fitnesses are bitwise equal, the hypothesis is dead and the strategy is back under
suspicion.

Two independent jitted functions are used deliberately: one returning only the new state
(byte-for-byte what `run.py` compiles) and one returning only the fitness. Adding an output
to a jitted function changes what XLA fuses, so asking both questions of one function would
risk measuring a program neither the driver nor the guard ever runs.
"""

import os
import subprocess
import sys

SHA = "74fa602"  # guard lowrank_r1, which the sweep ran and no test checked
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

env = {**os.environ, "PYTHONPATH": "src", "JAX_PLATFORMS": "cuda",
       "XLA_FLAGS": "--xla_gpu_deterministic_ops=true"}

run([sys.executable, "-c",
     "import jax; d=jax.devices(); print(jax.__version__, len(d), d[0].device_kind);"
     " assert len(d)==2, f'{len(d)} devices, want 2'"], env=env)

# Step 1. The invariance suite now includes lowrank_r1 (it did not when G1 was certified).
# This asks the same question at the reference problem's shape: a sphere with a (6,4) and a
# (4,) leaf, N=32. If lowrank_r1 fails here too the fault is broad; if it passes, whatever
# is happening needs the transformer_block shapes to show up.
print("\n== step 1: the invariance suite, now covering lowrank_r1 ==")
run([sys.executable, "-m", "pytest", "tests/gpu", "-m", "gpu", "-q", "-s"],
    env=env, check=False)

# Step 2. Reproduce the failing cell through the driver itself, both device counts in one
# process. check.py at this commit refuses mixed environments, so a pass here also confirms
# the directory is single-environment rather than the artefact that first misled us.
print("\n== step 2: the failing cell, through run.py, one process ==")
cfg = {
    "modes": ["strong"], "devices": [1, 2], "d_model": [256], "population": [64],
    "population_per_device": [32], "strategies": ["lowrank_r1", "iid_gaussian",
                                                  "mirrored_lr1"],
    "how": ["A", "B"], "batch": 8, "seq": 32, "warmup": 3, "repeats": 5,
    "matmul_precision": "highest", "results_dir": "results-lrdiag",
}
import json  # noqa: E402

with open("experiments/phase2/lrdiag.yaml", "w") as fh:
    json.dump(cfg, fh)  # YAML is a superset of JSON

run([sys.executable, "experiments/phase2/run.py",
     "--config", "experiments/phase2/lrdiag.yaml", "--hbm", "16", "--budget", "2400"],
    env=env)
run([sys.executable, "experiments/phase2/check.py",
     "--results", "experiments/phase2/results-lrdiag"], env=env, check=False)

# Step 3. The decisive measurement.
DIAG = r'''
import sys
import numpy as np
import jax

sys.path.insert(0, "experiments/phase2")
from shardes import sharding
from shardes.core import ShardedES
from shardes.strategies.registry import STRATEGIES
from shardes.problems import transformer_block
from shardes.shaping import centered_ranks

SEED, SIGMA, LR = 0, 0.01, 0.05
D_MODEL, N, BATCH, SEQ = 256, 64, 8, 32


def setup(devices, strategy, how):
    mesh = sharding.make_mesh(devices)
    key = jax.random.key(SEED)
    params = transformer_block.init(key, d_model=D_MODEL)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1),
                                         d_model=D_MODEL, batch=BATCH, seq=SEQ)
    es = ShardedES(STRATEGIES[strategy].build(), n=N, sigma=SIGMA, lr=LR,
                   mesh=mesh, how=how)
    return es, es.init(key, params), batch


def fitness_only(devices, strategy, how):
    """A jit whose sole output is the fitness, so nothing else constrains its fusion."""
    es, state, batch = setup(devices, strategy, how)

    @jax.jit
    def f(state):
        pert, state = es.ask(state)
        return es.apply(transformer_block.loss, state, pert)(batch)

    with jax.default_matmul_precision("highest"):
        return np.asarray(jax.device_get(f(state)), np.float64)


def params_only(devices, strategy, how):
    """Exactly what run.py compiles: one generation, state in, state out."""
    es, state, batch = setup(devices, strategy, how)

    @jax.jit
    def g(state):
        pert, s = es.ask(state)
        fit = es.apply(transformer_block.loss, s, pert)(batch)
        return es.tell(s, pert, fit)

    with jax.default_matmul_precision("highest"):
        out = g(state)
    return np.concatenate([np.asarray(x, np.float64).ravel()
                           for x in jax.tree.leaves(out.params)])


def report(strategy, how):
    print("=" * 72)
    print(f"{strategy} / how={how}   d={D_MODEL} N={N}")
    f1, f2 = fitness_only(1, strategy, how), fitness_only(2, strategy, how)

    same = np.array_equal(f1, f2)
    ulps = int((f1 != f2).sum())
    print(f"  raw fitness bitwise equal : {same}   ({ulps}/{N} members differ)")
    if not same:
        d = np.abs(f1 - f2)
        rel = d / np.maximum(np.abs(f2), 1e-30)
        print(f"  max |df|                  : {d.max():.6e}")
        print(f"  max |df|/|f|              : {rel.max():.6e}")

    # Exact ties are the thing to look at. `centered_ranks` sorts, so tied members get their
    # order from the sort's tie-breaking rather than from their fitness, and nothing requires
    # that to agree between two shapes on a GPU. On CPU-8 the low-rank strategies tie and
    # `iid_gaussian` does not, which is the same split as the guard failure: a rank-1
    # perturbation at sigma=0.01 moves a d=256 loss by less than its float32 ulp, so members
    # collide exactly.
    uniq = len(np.unique(f1))
    counts = np.unique(np.unique(f1, return_counts=True)[1], return_counts=True)
    print(f"  distinct fitness values   : {uniq}/{N}"
          f"   {'<-- ties present' if uniq < N else ''}")
    if uniq < N:
        biggest = int(np.unique(f1, return_counts=True)[1].max())
        print(f"  largest group sharing one fitness: {biggest}")
        print(f"  tie-group sizes: {dict(zip(counts[0].tolist(), counts[1].tolist()))}")

    r1, r2 = np.argsort(np.argsort(f1)), np.argsort(np.argsort(f2))
    swapped = int((r1 != r2).sum())
    print(f"  members whose RANK changed: {swapped}/{N}")
    gaps = np.diff(np.sort(f1))
    print(f"  smallest gap between adjacent fitnesses: {gaps.min():.6e}")
    if swapped:
        for i in np.where(r1 != r2)[0][:6]:
            print(f"    member {i:3d}: rank {r1[i]:3d} -> {r2[i]:3d}  "
                  f"f1={f1[i]:.17g} f2={f2[i]:.17g}")

    w1 = np.asarray(centered_ranks(jax.numpy.asarray(f1)))
    w2 = np.asarray(centered_ranks(jax.numpy.asarray(f2)))
    dw = np.linalg.norm(w1 - w2) / max(np.linalg.norm(w2), 1e-30)
    print(f"  shaped weights, L2 rel    : {dw:.6e}")

    p1, p2 = params_only(1, strategy, how), params_only(2, strategy, how)
    dp = np.linalg.norm(p1 - p2) / max(np.linalg.norm(p2), 1e-30)
    print(f"  resulting params, L2 rel  : {dp:.6e}"
          f"   {'<-- the guard failure' if dp > 1e-5 else ''}")


# lowrank_r1 is the one that fails; the other two are controls that passed the same guard
# on the same node, so a difference between them and it is the signal.
for strategy in ("lowrank_r1", "iid_gaussian", "mirrored_lr1"):
    for how in ("A", "B"):
        report(strategy, how)

print("=" * 72)
print("Reading it:")
print("  fitness differs + ranks change  -> the shaping amplified a kernel-selection ulp;")
print("                                     the strategy is not at fault.")
print("  fitness identical + ranks change-> the sort broke exact ties differently at the")
print("                                     two shapes; the fix is a deterministic")
print("                                     tie-break in shaping, not in the strategy.")
print("  fitness identical + ranks same  -> the divergence is inside the strategy after")
print("                                     all, and this whole hypothesis is dead.")
'''

print("\n== step 3: does the fitness itself differ across device counts? ==")
with open("lrdiag_inner.py", "w") as fh:
    fh.write(DIAG)
run([sys.executable, "lrdiag_inner.py"], env=env, check=False)

print("\nlrdiag done")
