"""Evolution strategies on eight devices, on a laptop.

    python examples/quickstart.py

The README shows this file, and `tests/test_examples.py` runs it, so the README cannot
drift from what works. Eight *simulated* CPU devices stand in for accelerators: the
sharding is the same program either way, which is how this library is tested.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"   # 8 devices on a CPU

import jax
import shardes
from shardes.problems import transformer_block       # a small model that uses the seams

key = jax.random.key(0)
params = transformer_block.init(key, d_model=64)
batch = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=64, batch=8, seq=32)

mesh = shardes.make_mesh()                           # every visible device, one "pop" axis
es = shardes.ShardedES(shardes.Mirrored(shardes.LowRank(r=1)),
                       n=256, sigma=1e-2, lr=1e-2, mesh=mesh)
state = es.init(key, params)

@jax.jit
def generation(state):
    pert, state = es.ask(state)                      # a Perturbation, never a batch of params
    fitness = es.apply(transformer_block.loss, state, pert)(batch)   # one loss per member
    return es.tell(state, pert, fitness), fitness

for step in range(30):
    state, fitness = generation(state)
    if step % 10 == 0 or step == 29:
        print(f"step {step:2d}  mean loss {float(fitness.mean()):.4f}")
