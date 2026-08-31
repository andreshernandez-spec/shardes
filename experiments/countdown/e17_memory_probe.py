#!/usr/bin/env python
"""Why the low-rank arms OOM in E17b and the seed arm does not.

    python e17_memory_probe.py

E17b records 57 OOM cells. Every one is a low-rank arm; `mirrored_seed` fits
every cell of the grid. E17's README read that as the storage-for-compute
trade (factors cost memory), and e17b.yaml froze the prediction that rank 16
would OOM where rank 1 runs. Both are wrong, and the grid itself says so: the
three ranks OOM at exactly the same cells, with HBM temporaries agreeing to
0.1% (114.58 / 114.61 / 114.66 G at N=128, D=1). Memory that does not move
with r is not the factors.

This prices the three candidates by compiling one generation on simulated
devices and reading XLA's own temporary estimate, sweeping the thing each
hypothesis says should move it:

  factors           -> scales with r
  perturbed weights -> scales with members per device, not with the batch
  activations       -> scales with members per device AND with the batch

Vocabulary is kept at Qwen's 151936 because the logits are the term in
question; everything else is small, so this runs on a laptop CPU. The
absolute numbers are CPU numbers and mean nothing. The scalings are the
result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_es import STRATEGIES  # noqa: E402

import harness  # noqa: E402
from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import qwen2  # noqa: E402

# Qwen2.5-0.5B's vocabulary and head, everything else cut so this fits a
# laptop. The logits are (members, prompts, T-1, vocab), which is the term
# under test, so vocab is the one dimension that must stay honest.
CFG = qwen2.Config(vocab=151936, d_model=64, n_layers=2, n_heads=4,
                   n_kv_heads=2, d_ff=128)


def temporaries(strategy: str, members: int, devices: int, prompts: int,
                pad_to: int) -> float:
    """GiB of temporaries XLA reserves for one compiled generation."""
    mesh = sharding.make_mesh(devices)
    params = qwen2.init(jax.random.key(0), CFG, dtype=jnp.bfloat16)
    ids = jax.random.randint(jax.random.key(1), (prompts, pad_to), 0, CFG.vocab)
    batch = (ids, jnp.ones_like(ids))
    es = ShardedES(STRATEGIES[strategy]({}), n=members * devices, sigma=1e-3,
                   lr=1.0, mesh=mesh, how="A", compute_dtype=jnp.bfloat16)
    state = es.init(jax.random.key(7), params)

    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(lambda p, b: qwen2.nll(p, b, CFG), state, pert)(batch)
        return es.tell(state, pert, fitness)

    an = jax.jit(generation).lower(state).compile().memory_analysis()
    return an.temp_size_in_bytes / 2**30


def main() -> None:
    print(harness.capture_env(HERE, ())["commit"][:12], "on",
          f"{len(jax.devices())} simulated {jax.devices()[0].platform} devices")
    arms = ["mirrored_seed", "mirrored_lr1", "mirrored_lr4", "mirrored_lr16"]

    print("\nA. members per device, at 8 prompts of 128 tokens, D=1")
    print(f"{'members':>8} " + " ".join(f"{a.replace('mirrored_', ''):>14}"
                                        for a in arms))
    for members in (2, 4, 8, 16):
        row = [temporaries(a, members, 1, 8, 128) for a in arms]
        print(f"{members:>8} " + " ".join(f"{g:>13.3f}G" for g in row))

    print("\nB. prompts in the batch, at 4 members per device, D=1")
    print(f"{'prompts':>8} " + " ".join(f"{a.replace('mirrored_', ''):>14}"
                                        for a in arms))
    for prompts in (2, 4, 8, 16):
        row = [temporaries(a, 4, 1, prompts, 128) for a in arms]
        print(f"{prompts:>8} " + " ".join(f"{g:>13.3f}G" for g in row))

    print("\nC. rank, at 8 members per device, 8 prompts, D=1")
    for arm in arms:
        print(f"  {arm.replace('mirrored_', ''):>6}: "
              f"{temporaries(arm, 8, 1, 8, 128):.3f}G")

    print("\nD. seed with the low-rank arms' evaluation (chunk = members)")
    print("   the arms differ by chunking, not by perturbation scheme, so")
    print("   un-chunking the seed arm should reproduce the low-rank curve")
    print(f"{'members':>8} {'seed chunk=1':>14} {'seed chunk=n':>14} {'lr1':>14}")
    for members in (2, 4, 8, 16):
        chunked = temporaries_chunked(members)
        print(f"{members:>8} "
              f"{temporaries('mirrored_seed', members, 1, 8, 128):>13.3f}G "
              f"{chunked:>13.3f}G "
              f"{temporaries('mirrored_lr1', members, 1, 8, 128):>13.3f}G")


def temporaries_chunked(members: int) -> float:
    """The seed arm evaluating all its members in one vmap, as LowRank does."""
    from shardes.strategies.mirrored import Mirrored
    from shardes.strategies.seed_regenerated import SeedRegenerated

    mesh = sharding.make_mesh(1)
    params = qwen2.init(jax.random.key(0), CFG, dtype=jnp.bfloat16)
    ids = jax.random.randint(jax.random.key(1), (8, 128), 0, CFG.vocab)
    batch = (ids, jnp.ones_like(ids))
    # Mirrored splits the members into a plus and a minus side, so the scan
    # underneath sees members/2 and that is the chunk that un-chunks it.
    es = ShardedES(Mirrored(SeedRegenerated(chunk=max(1, members // 2))),
                   n=members, sigma=1e-3, lr=1.0, mesh=mesh, how="A",
                   compute_dtype=jnp.bfloat16)
    state = es.init(jax.random.key(7), params)

    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(lambda p, b: qwen2.nll(p, b, CFG), state, pert)(batch)
        return es.tell(state, pert, fitness)

    an = jax.jit(generation).lower(state).compile().memory_analysis()
    return an.temp_size_in_bytes / 2**30


if __name__ == "__main__":
    main()
