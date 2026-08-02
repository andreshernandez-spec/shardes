"""Does the Phase 2 sweep actually run on a TPU v5e-8? Answered before a session is spent.

The queue for this tier measured ~2.5 h on 2026-08-01, so T1 is one shot per attempt. This
kernel is the cheap shot: it proves the three things that could take the real run out, in
the order they would fail, and does nothing else.

1. **The jax upgrade.** The image ships jax 0.10.2 and `pyproject.toml` floors at 0.11.
   `jax[tpu]` is coupled to `libtpu` in a way `jax[cuda12]` is not, so an upgrade that works
   on GPU is not evidence here. If this takes the runtime out, 8 devices become 0 and every
   later step is meaningless, which is why it is checked first and on its own.
2. **The library on TPU.** `shard_map`, `psum` and `make_mesh` have only ever run on CPU and
   CUDA in this project. Nothing about the code is GPU-specific, but "should port" is not a
   measurement.
3. **The driver end to end**, at rehearsal size, on 8 real chips. If the sweep is going to
   die on config 1, it should die here for free.

Deliberately NOT run here: `tests/gpu`. Those tests filter on `platform in (gpu, cuda,
rocm)`, so on a TPU they all skip and report green. Reading that as a pass would be the same
mistake as reading a silent accelerator downgrade as a pass.
"""

import os
import subprocess
import sys

SHA = "2db532a"  # SET THIS to the commit under test.
REPO = "https://github.com/andreshernandez-spec/shardes.git"
CHECKOUT = "/kaggle/working/shardes"


def run(cmd, check=True, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, text=True, **kw)
    if check and r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r


print("== before the upgrade ==")
run([sys.executable, "-c", "import jax; print(jax.__version__, jax.device_count())"],
    check=False)

# Step 1. The upgrade, and then a *fresh interpreter* to see it: the running process already
# imported the old jax and initialized the TPU backend.
run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]>=0.11"])

print("\n== step 1: does the TPU runtime survive the upgrade? ==")
run([
    sys.executable, "-c",
    "import jax; v=jax.__version__; d=jax.devices();"
    " print(v, len(d), d[0].platform, d[0].device_kind);"
    " assert tuple(map(int, v.split('.')[:2])) >= (0, 11), f'jax {v} < 0.11';"
    " assert len(d) == 8, f'{len(d)} devices after the upgrade, want 8';"
    " assert d[0].platform == 'tpu', f'platform is {d[0].platform}, not tpu'",
])

run(["git", "clone", "-q", REPO, CHECKOUT])
os.chdir(CHECKOUT)
run(["git", "checkout", "-q", SHA])
run(["git", "log", "--oneline", "-1"])

env = {**os.environ, "PYTHONPATH": "src"}

print("\n== step 2: shard_map, psum and make_mesh on 8 real chips ==")
run([
    sys.executable, "-c",
    "import jax, jax.numpy as jnp;"
    " from shardes import sharding;"
    " from shardes.core import ShardedES;"
    " from shardes.strategies.iid_gaussian import IIDGaussian;"
    " mesh = sharding.make_mesh(8);"
    " es = ShardedES(IIDGaussian(), n=32, sigma=0.01, lr=0.05, mesh=mesh, how='B');"
    " p = {'w': jnp.ones((8, 8), jnp.float32)};"
    " s = es.init(jax.random.key(0), p);"
    " f = lambda q, _x: sum(jnp.sum(jnp.square(l)) for l in jax.tree.leaves(q));"
    " pert, s2 = es.ask(s);"
    " fit = es.apply(f, s2, pert)(jnp.zeros(()));"
    " out = es.tell(s2, pert, fit);"
    " jax.block_until_ready(out);"
    " print('one generation OK, params norm', float(jnp.linalg.norm(out.params['w'])))",
], env=env)

print("\n== step 3: the phase 2 driver, rehearsal size, on 8 chips ==")
run([sys.executable, "experiments/phase2/run.py",
     "--config", "experiments/phase2/rehearsal.yaml", "--budget", "900"], env=env)

print("\n== the guard ==")
result = run([sys.executable, "experiments/phase2/check.py",
              "--results", "experiments/phase2/results-rehearsal"], env=env, check=False)

print(f"\ntpuprep exit code: {result.returncode}")
sys.exit(result.returncode)
