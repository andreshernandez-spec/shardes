#!/usr/bin/env python
"""B1: why does scrambled Sobol degrade with N? A discriminating experiment.

    python sobol_b1.py                  # the table
    python sobol_b1.py --quick          # smaller, for a smoke test

E1 measured `mirrored_sobol_lr1` as systematically worse than uncoupled `mirrored_lr1`, with
the gap growing monotonically in N (0.988 at 2^14, 0.966 at 2^16, 0.892 at 2^18). Lost design
diversity and marginal-moment error were both ruled out with numbers (`docs/BACKLOG.md` B1).
This asks whether the cause is the construction rather than the method.

---

**The hypothesis, stated precisely enough to be wrong.**

Every stream — one per (leaf, factor axis), so 12 for a 6-leaf rank-1 model — calls
`sobol_point` with the *same* direction numbers and its own random digital shift. For two
members `i`, `j` within a stream:

    (x_i XOR sigma) XOR (x_j XOR sigma)  =  x_i XOR x_j

The shift cancels. So the inter-member XOR geometry is **identical in every stream**, merely
translated, where i.i.d. gives each stream an independent arrangement. If that one shared
arrangement has a deficiency, it is added 12 times coherently instead of being averaged over
12 independent draws.

**The discriminating prediction.** Penalty should scale with the *number of streams*, which
is what varying the leaf count changes. If instead the penalty is flat in leaf count, the
cause is not coherence and the two a-priori reasons recorded in `docs/01` C0.5 (high effective
dimension, degraded 2-D projections past a few hundred dimensions) are the explanation. That
is a property of Sobol at this dimension and becomes a documented limitation rather than a bug.

**And a direct test of the fix.** `offset` gives each stream a disjoint block of Sobol
dimensions instead of the same block shifted, which breaks the shared geometry at the source.
`permute` keeps the same dimensions but permutes which one feeds which coordinate, per stream.
If either removes the penalty, the hypothesis is confirmed *and* the fix is in hand. If both
leave it, the construction is exonerated.

Oracle is backprop, as in Phase 0: a differentiable model, so `cos(g_hat, grad f)` is measured
against the real thing rather than against a reference estimator.
"""

from __future__ import annotations

import argparse
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from shardes import metrics, shaping
from shardes.coupling import _SOBOL_BITS, _UNIFORM_BITS, Gaussian, _direction_numbers, sobol_point
from shardes.estimator import estimate
from shardes.nn import dense
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.types import Array, Key


class SobolVariant:
    """Scrambled Sobol with a per-stream decorrelation knob.

    `mode`:
      "shared"  — what ships. Same direction numbers everywhere, shift per stream.
      "offset"  — a disjoint block of Sobol dimensions per stream. Breaks the shared XOR
                  geometry at its source, at the cost of using higher dimensions, whose
                  projections are worse. Both effects are real and this cannot separate them,
                  which is why `permute` exists too.
      "permute" — same dimensions, but a per-stream permutation of which dimension feeds
                  which coordinate. Keeps the dimension quality identical and still moves a
                  given dimension's deficiency onto a different coordinate in each stream.
    """

    def __init__(self, mode: str = "shared", max_dim: int = 8192):
        self.mode = mode
        self.max_dim = max_dim

    def __call__(self, stream: Key, member_id: Array, d: int, dtype) -> Array:
        k_shift, k_off, k_perm = jax.random.split(stream, 3)
        shift = jax.random.bits(k_shift, (d,), jnp.uint32) >> (32 - _SOBOL_BITS)

        if self.mode == "offset":
            # Direction numbers are (bits, d), so a block of dimensions is a slice of
            # *columns*, not rows. The block index is traced, hence dynamic_slice.
            blocks = max(self.max_dim // d, 1)
            b = jax.random.randint(k_off, (), 0, blocks)
            table = jnp.asarray(_direction_numbers(blocks * d))
            v = jax.lax.dynamic_slice_in_dim(table, b * d, d, axis=1)
        else:
            v = jnp.asarray(_direction_numbers(d))

        x = sobol_point(v, member_id, shift)
        if self.mode == "permute":
            x = x[jax.random.permutation(k_perm, d)]

        u = (x >> (_SOBOL_BITS - _UNIFORM_BITS)).astype(jnp.float32)
        return jax.scipy.special.ndtri((u + 0.5) * 2.0**-_UNIFORM_BITS).astype(dtype)


def make_model(n_leaves: int, m: int, k: int, key: Key):
    """`n_leaves` independent (m, k) weights summed into one prediction.

    Leaf count is the only thing that varies, so it is the only thing the penalty can be
    scaling with. Every leaf has the same shape, so `d_eff`, the ambient dimension and the
    per-leaf conditioning all move together with it and none of them moves independently.
    """
    keys = jax.random.split(key, n_leaves + 2)
    params = {f"w{i}": jax.random.normal(keys[i], (m, k), jnp.float32) / jnp.sqrt(float(k))
              for i in range(n_leaves)}
    x = jax.random.normal(keys[-2], (16, k), jnp.float32)
    y = jax.random.normal(keys[-1], (16, m), jnp.float32)

    def model(p, _batch):
        pred = sum(dense(x, p[f"w{i}"]) for i in range(n_leaves))
        return 0.5 * jnp.mean(jnp.square(pred - y))

    return params, model


def quality(coupling, params, model, n: int, sigma: float, replicates: int, key: Key):
    """(mean, standard error) of cos(g_hat, grad f) over replicates. Higher mean is better.

    The standard error is not decoration. The effects here are a couple of percent, and
    without it there is no way to tell a real trend from R being too small — which is exactly
    the mistake made twice already in this project by calling a trend from partial data.
    """
    truth = jax.grad(model)(params, None)
    strategy = Mirrored(LowRank(r=1, coupling=coupling))
    ids = jnp.arange(n)

    def one(k):
        g = estimate(strategy, model, params, None, k, member_ids=ids, sigma=sigma,
                     shaping=shaping.none)
        return metrics.cosine_similarity(g, truth)

    # lax.map, not vmap: vmapping the replicate axis multiplies the whole forward pass and
    # OOMs a 16 GB card at these sizes. It is the activations that bind, not the perturbation
    # (docs/01 C0.2), so the fix is to run replicates in sequence at the memory of one.
    cos = jax.lax.map(one, jax.random.split(key, replicates))
    return float(jnp.mean(cos)), float(jnp.std(cos) / jnp.sqrt(replicates))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    # Sized for CPU. The question is whether the penalty scales with the *number of
    # streams*, and that needs replicates for statistical power far more than it needs a
    # large m or k -- the effects are a couple of percent, so R is the binding constraint.
    m = k = 64
    leaves = (1, 2, 4) if args.quick else (1, 2, 4, 8)
    n = 1024 if args.quick else 2048
    replicates = 8 if args.quick else 64
    sigma = 0.01

    variants = {
        "iid": Gaussian(),
        "sobol/shared": SobolVariant("shared"),
        "sobol/offset": SobolVariant("offset"),
        "sobol/permute": SobolVariant("permute"),
    }

    print(f"leaves x ({m},{k}) each, N={n}, sigma={sigma}, R={replicates}, rank 1, mirrored")
    print("cos(g_hat, grad f), and each sobol variant as a ratio to iid\n")
    print(f"{'leaves':>7}{'streams':>9}{'iid cos':>17}" +
          "".join(f"{v.split('/')[-1] + ' ratio':>15}" for v in list(variants)[1:]))
    print("-" * (16 + 17 + 15 * (len(variants) - 1)))

    rows = {}
    for n_leaves in leaves:
        params, model = make_model(n_leaves, m, k, jax.random.key(0))
        got = {
            name: quality(c, params, model, n, sigma, replicates, jax.random.key(1))
            for name, c in variants.items()
        }
        rows[n_leaves] = got
        base, base_se = got["iid"]
        cells = ""
        for v in variants:
            mu, se = got[v]
            if v == "iid":
                cells += f"{mu:>9.4f}+-{se:.4f}"
            else:
                # ratio and its standard error, propagated
                r = mu / base
                r_se = r * ((se / mu) ** 2 + (base_se / base) ** 2) ** 0.5
                cells += f"{r:>8.3f}+-{r_se:.3f}"
        print(f"{n_leaves:>7}{2 * n_leaves:>9}{cells}")

    print("""
Read the *trend down the columns*, not the absolute numbers.

  sobol/shared ratio falling as leaves grow  -> the penalty scales with the number of
      streams, which is what the shared XOR geometry predicts. The construction is at fault
      and `offset`/`permute` should recover it.
  sobol/shared ratio flat in leaves          -> not coherence. The cause is Sobol itself at
      this dimension, and B1 closes as a documented limitation rather than a bug.
  offset or permute recovering the ratio     -> the fix, directly demonstrated.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
