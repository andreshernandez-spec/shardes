#!/usr/bin/env python
"""Evidence for the bf16 dtype-policy decision. CPU, seconds, no GPU.

    python probe.py

Three questions, each answered by running code rather than by arithmetic on paper,
because the last dtype argument this project accepted on paper was wrong for three
documents (docs/postmortem-gpu-determinism.md).

1. What does `tell` do to a bf16 params tree today?
2. How much of an intended parameter movement survives being accumulated in bf16,
   as a function of the per-step size? This is the case for master weights.
3. How many distinct fitness values does a real population have in bf16? Ranks are
   computed from fitness, so ties are where `centered_ranks` degrades. This is the
   case for keeping fitness out of bf16 whatever the params policy says.

The numbers quoted in docs/proposal-bf16-policy.md come from this script at the commit
recorded there. If the library's dtype handling changes, rerun and update the doc.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "cpu")

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import transformer_block  # noqa: E402
from shardes.strategies.iid_gaussian import IIDGaussian  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402


def q1_dtype_drift():
    """bf16 in, what out? One ask/apply/tell cycle per strategy."""
    print("Q1: params dtype after one tell (input bf16)")
    mesh = sharding.make_mesh(1)
    for name, strat in (
        ("iid_gaussian", IIDGaussian()),
        ("mirrored_seed", Mirrored(SeedRegenerated())),
    ):
        es = ShardedES(strat, n=8, sigma=0.01, lr=0.05, mesh=mesh)
        params = {"w": jnp.full((4, 4), 0.05, dtype=jnp.bfloat16)}
        st = es.init(jax.random.key(0), params)
        pert, st = es.ask(st)
        fit = es.apply(lambda p, x: jnp.sum(p["w"] * x), st, pert)(
            jnp.ones((4, 4), jnp.bfloat16)
        )
        st2 = es.tell(st, pert, fit)
        print(f"  {name:14} params: bfloat16 -> {st2.params['w'].dtype}, "
              f"fitness: {fit.dtype}")
    print()


def q2_update_survival():
    """Apply T small SGD steps to bf16 weights vs an f32 master, compare movement.

    Steps are random signed, scaled to a fixed size relative to the weight magnitude,
    the shape an ES update takes after `lr/(n*sigma)`. The metric is how much of the
    f32 master's total movement the bf16 accumulator reproduced.

    bf16 has 8 significand bits, so the relative ulp is 2^-8 = 3.9e-3. Steps well
    below that round to nothing; steps near it survive by luck of alignment.
    """
    print("Q2: fraction of intended movement surviving bf16 accumulation, T=200 steps")
    key = jax.random.key(1)
    w0 = 0.05  # the transformer block's init scale at d=512 is 1/sqrt(512) ~ 0.044
    for rel in (1e-2, 3.9e-3, 1e-3, 1e-4):
        k = jax.random.fold_in(key, hash(rel) % (2**31))
        steps = jax.random.normal(k, (200,), jnp.float32) * (rel * w0)
        wb = jnp.asarray(w0, jnp.bfloat16)
        wf = jnp.asarray(w0, jnp.float32)
        for s in steps:
            wb = (wb + s.astype(jnp.bfloat16)).astype(jnp.bfloat16)
            wf = wf + s
        moved_b = float(wb.astype(jnp.float32) - w0)
        moved_f = float(wf - w0)
        print(f"  step {rel:7.1e} of |w|:  bf16 moved {moved_b:+.2e}, "
              f"f32 moved {moved_f:+.2e}, survival {moved_b / moved_f:6.1%}")
    print()


def q3_fitness_ties():
    """A real population's fitness, in f32 and cast to bf16: how many distinct values?

    d=64, n=256 on the CPU transformer block. The absolute numbers move with the
    shape; the collapse from f32 to bf16 is the point.
    """
    print("Q3: distinct fitness values in a population of 256 (d=64 block)")
    mesh = sharding.make_mesh(1)
    es = ShardedES(IIDGaussian(), n=256, sigma=0.01, lr=0.05, mesh=mesh)
    key = jax.random.key(2)
    params = transformer_block.init(key, d_model=64)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=64)
    st = es.init(key, params)
    pert, st = es.ask(st)
    fit = es.apply(transformer_block.loss, st, pert)(batch)
    f32 = jnp.unique(fit).size
    b16 = jnp.unique(fit.astype(jnp.bfloat16)).size
    spread = float(jnp.max(fit) - jnp.min(fit))
    print(f"  spread {spread:.3e}; distinct in f32: {f32}/256, in bf16: {b16}/256")
    print()


if __name__ == "__main__":
    q1_dtype_drift()
    q2_update_survival()
    q3_fitness_ties()
