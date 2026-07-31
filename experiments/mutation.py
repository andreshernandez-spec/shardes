#!/usr/bin/env python
"""Mutation testing: break the library on purpose and check a test notices.

    python mutation.py                 # every mutation
    python mutation.py -k sobol        # a subset, by id

A test that has never failed is not obviously a test. Each entry below is a small, *plausible*
edit — the kind a refactor or a tired afternoon would produce — paired with the tests that
should catch it. A **survivor** is a mutation nothing caught, and it names a real gap.

Deliberately hand-written rather than generated. A generic mutator spends most of its budget
on edits that are unreachable, equivalent, or obviously silly, and the interesting question
here is not "is every line covered" but "does the suite defend the *invariants*" — the seed
contract, unit scale, additivity, the sign convention. Those need edits chosen by someone who
knows what they are.

Two things learned the hard way, both of which silently produced wrong results before:

- **Mutations replace, they never append.** An earlier harness appended after a `return`,
  so the edit was unreachable and every mutation "survived", which reads as a catastrophic
  test suite rather than a broken harness.
- **`PYTHONDONTWRITEBYTECODE=1`.** CPython's `.pyc` cache keys on (mtime, size). Two
  mutations written in the same second with the same length hit the cache, so a later case
  ran an earlier one's bytecode — producing a *wrong attribution*, which is worse than a
  crash because it looks like data.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "shardes"


@dataclasses.dataclass(frozen=True)
class Mutation:
    id: str
    path: str
    old: str
    new: str
    tests: str
    why: str


MUTATIONS = [
    # --- the seed contract -------------------------------------------------------------
    Mutation(
        "noise/position-not-id", "strategies/_noise.py",
        "coupling(s, member_id, x.size, x.dtype).reshape(x.shape)",
        "coupling(s, jnp.int32(0), x.size, x.dtype).reshape(x.shape)",
        "tests/test_strategies.py tests/test_core.py",
        "every member gets member 0's noise: the seed contract collapsed",
    ),
    # --- unit scale --------------------------------------------------------------------
    Mutation(
        "lowrank/drop-sqrt-r", "strategies/lowrank.py",
        "return scale * jnp.einsum(\"n,nmr,nkr->mk\", w, a, lf.b.astype(jnp.float32))",
        "return jnp.einsum(\"n,nmr,nkr->mk\", w, a, lf.b.astype(jnp.float32))",
        "tests/test_lowrank.py tests/test_strategies.py",
        "rank-r perturbation no longer unit scale; r=1 unaffected, so r=4 must catch it",
    ),
    Mutation(
        "hd/wrong-normalisation", "coupling.py",
        "return (v[:d] / n ** ((self.factors - 1) / 2)).astype(dtype)",
        "return (v[:d] / n ** (self.factors / 2)).astype(dtype)",
        "tests/test_coupling.py tests/test_strategies.py",
        "HD rows off by sqrt(n): second moment no longer 1",
    ),
    # --- mirroring ---------------------------------------------------------------------
    Mutation(
        "mirrored/sum-not-difference", "strategies/mirrored.py",
        "return self.inner.contract(pert.inner, weights[0::2] - weights[1::2])",
        "return self.inner.contract(pert.inner, weights[0::2] + weights[1::2])",
        "tests/test_mirrored.py tests/test_estimator.py",
        "antithetic pairs add instead of cancelling: the estimator points nowhere",
    ),
    Mutation(
        "mirrored/no-sign-flip", "strategies/mirrored.py",
        "negative = self.inner.apply(model, params, pert.inner,\n"
        "                                    jax.tree.map(lambda s: -s, sigma))",
        "negative = self.inner.apply(model, params, pert.inner, sigma)",
        "tests/test_mirrored.py tests/test_estimator.py",
        "both halves evaluated at +sigma: mirroring becomes duplication",
    ),
    # --- shaping -----------------------------------------------------------------------
    Mutation(
        "centered/drop-correction", "shaping.py",
        "return (fitness - jnp.mean(fitness)) * (n / (n - 1))",
        "return fitness - jnp.mean(fitness)",
        "tests/test_estimator.py",
        "the (1 - 1/n) bias returns; this is the trap Phase 0 measured",
    ),
    # `sd + 1e-8` instead of the `where` was tried here and is an **equivalent mutant**, not
    # a gap: for a task nobody varies on the numerator is zero too, and for a nearly-dead task
    # standardisation is scale-invariant, so |z| <= sqrt(n-1) whatever the spread. Measured:
    # weights stay at ~1.5 for spreads from 1e-2 down to 1e-7. Recorded so it is not retried.
    Mutation(
        "group_relative/drop-std", "shaping.py",
        "advantage = jnp.where(sd > 0, (fitness - mu) / jnp.where(sd > 0, sd, 1.0), 0.0)",
        "advantage = jnp.where(sd > 0, fitness - mu, 0.0)",
        "tests/test_shaping.py",
        "centering without standardising: a task's reward *scale* drives the update again, "
        "which is the one thing group-relative shaping exists to prevent",
    ),
    # --- the estimator -----------------------------------------------------------------
    Mutation(
        "estimator/sign-flip", "estimator.py",
        "return jax.tree.map(lambda u: u / (n * sigma), update)",
        "return jax.tree.map(lambda u: -u / (n * sigma), update)",
        "tests/test_estimator.py tests/test_core.py",
        "descends into ascent: trains smoothly the wrong way",
    ),
    # --- transforms --------------------------------------------------------------------
    Mutation(
        "fwht/butterfly-swap", "transforms/fwht.py",
        "x = jnp.stack([a + b, a - b], axis=-2).reshape(*batch, n)",
        "x = jnp.stack([a - b, a + b], axis=-2).reshape(*batch, n)",
        "tests/test_fwht.py tests/test_coupling.py",
        "butterfly halves swapped: no longer the Hadamard transform",
    ),
    Mutation(
        "hadamard_row/drop-parity", "coupling.py",
        "parity = jax.lax.population_count(j & p.astype(jnp.uint32)) & jnp.uint32(1)",
        "parity = jax.lax.population_count(j & p.astype(jnp.uint32))",
        "tests/test_coupling.py",
        "parity without the mask: rows are no longer +/-1",
    ),
    # --- sobol, including the B1 fix ---------------------------------------------------
    # Every stream draws block 0, which is the pre-B1 defect *with the shapes still valid*.
    # An earlier version disabled the slice entirely, which left `v` at the full span width
    # and was "caught" by a TypeError. A mutation that crashes proves the harness ran, not
    # that the suite defends anything, so it has to stay well-formed.
    Mutation(
        "sobol/undo-b1-blocks", "coupling.py",
        "b = jax.random.randint(k_block, (), 0, self._blocks(d))",
        "b = jnp.int32(0)",
        "tests/test_coupling.py",
        "every stream shares one block again: the defect B1 fixed",
    ),
    Mutation(
        "sobol/no-scramble", "coupling.py",
        "jax.random.bits(k_shift, (d,), jnp.uint32) >> (32 - _SOBOL_BITS)",
        "jnp.zeros((d,), jnp.uint32)",
        "tests/test_coupling.py tests/test_estimator.py",
        "deterministic sobol: the estimator is a fixed vector and biased at fixed N",
    ),
    # --- the linear seam ---------------------------------------------------------------
    Mutation(
        "nn/ignore-structured", "nn.py",
        "    if isinstance(w, StructuredWeight):\n        return w.apply_to(x)\n    return x @ w.T",
        "    return x @ w.T",
        "tests/test_nn.py tests/test_lowrank.py",
        "structured weights stop dispatching: LowRank silently wrong or raising",
    ),
    # --- the diagonal ------------------------------------------------------------------
    # Broadcasts leaf 0 everywhere: a scalar sigma is unaffected, a diagonal collapses. The
    # earlier version referenced `jnp`, which _scale.py does not import, and was "caught" by a
    # NameError -- see the note on sobol/undo-b1-blocks.
    Mutation(
        "scale/ignore-tree", "strategies/_scale.py",
        "    if jax.tree.structure(sigma) == jax.tree.structure(like):\n        return sigma\n"
        "    return jax.tree.map(lambda _: sigma, like)",
        "    first = jax.tree.leaves(sigma)[0]\n    return jax.tree.map(lambda _: first, like)",
        "tests/test_core.py",
        "a per-coordinate diagonal collapses to one leaf's value: the feature silently "
        "does nothing while a scalar sigma still works, so only a diagonal test can see it",
    ),
    # --- phase 1: sharding ---------------------------------------------------------------
    Mutation(
        "sharding/local-ids-not-global", "sharding.py",
        "    return jax.device_put(jnp.arange(n, dtype=jnp.int32), members(mesh))",
        "    per = n // n_devices(mesh)\n"
        "    local = jnp.tile(jnp.arange(per, dtype=jnp.int32), n_devices(mesh))\n"
        "    return jax.device_put(local, members(mesh))",
        "tests/test_sharding.py tests/test_core.py",
        "each device numbers its members from 0: the seed contract breaks and every device "
        "draws the same perturbations. This is THE bug the whole design exists to prevent",
    ),
    Mutation(
        "sharding/skip-even-split-check", "sharding.py",
        "    if n % d:\n        raise ValueError(",
        "    if False:\n        raise ValueError(",
        "tests/test_sharding.py tests/test_core.py",
        "an uneven population reaches shard_map, where it changes the update rather than "
        "failing",
    ),
    Mutation(
        "sharding/skip-pair-check", "sharding.py",
        "    if paired and (n // d) % 2:",
        "    if False and (n // d) % 2:",
        "tests/test_sharding.py tests/test_core.py",
        "a mirrored pair straddles a device boundary and the antithetic cancellation is lost "
        "silently: the run still produces an update, just a worse one",
    ),
    # --- phase 1: contraction ------------------------------------------------------------
    Mutation(
        "contraction/drop-psum", "contraction.py",
        "        return jax.tree.map(lambda leaf: jax.lax.psum(leaf, POP), partial)",
        "        return jax.tree.map(lambda leaf: jax.lax.pmean(leaf, POP), partial)",
        "tests/test_contraction.py tests/test_core.py",
        "Strategy B scales its local partial instead of summing across devices. Correct at "
        "D=1 and at every D if the shards happened to be equal, so only a real multi-device "
        "comparison can see it",
    ),
    Mutation(
        "contraction/A-forgets-to-gather-ids", "contraction.py",
        "    ids = jax.lax.with_sharding_constraint(member_ids, rep)",
        "    ids = member_ids",
        "tests/test_contraction.py tests/test_core.py",
        "Strategy A regenerates from un-gathered ids: each device contracts only its own "
        "members but does not psum, so the update is a fraction of the truth",
    ),
    # --- phase 1: the core ---------------------------------------------------------------
    Mutation(
        "core/reuse-the-generation-key", "core.py",
        "        base_key = jax.random.fold_in(state.key, state.generation)",
        "        base_key = state.key",
        "tests/test_core.py",
        "every generation samples the same population: training stalls in a way that looks "
        "like a bad learning rate",
    ),
    Mutation(
        "core/tell-ascends", "core.py",
        "                lambda p, s, u: p - (self.lr / (self.n * s)) * u,",
        "                lambda p, s, u: p + (self.lr / (self.n * s)) * u,",
        "tests/test_core.py",
        "tell ascends on the objective: trains smoothly in the wrong direction",
    ),
    Mutation(
        "core/drop-sigma-from-tell", "core.py",
        "                lambda p, s, u: p - (self.lr / (self.n * s)) * u,",
        "                lambda p, s, u: p - (self.lr / self.n) * u,",
        "tests/test_core.py",
        "the 1/sigma normalisation goes: g_hat estimates sigma*grad f, so the step size "
        "silently scales with the exploration radius",
    ),
    # --- dimensions --------------------------------------------------------------------
    Mutation(
        "dimensions/full-for-lowrank", "dimensions.py",
        "            total += int(rank) * int(sum(leaf.shape))",
        "            total += int(leaf.size)",
        "tests/test_dimensions.py",
        "d_eff reports the ambient dimension at every rank: F5's x-axis stops describing "
        "what was sampled, and the rank-1 panel silently becomes the full-rank one",
    ),
]


def run(mutation: Mutation, timeout: float) -> tuple[bool, str]:
    """Apply, test, restore. Returns (caught, detail)."""
    path = SRC / mutation.path
    original = path.read_text()
    if mutation.old not in original:
        return False, "STALE: the code no longer contains the text this mutates"
    if original.count(mutation.old) != 1:
        return False, f"AMBIGUOUS: {original.count(mutation.old)} matches"

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu"}
    try:
        path.write_text(original.replace(mutation.old, mutation.new))
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider",
             *mutation.tests.split()],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return True, "caught (timeout — an infinite loop counts as noticing)"
    finally:
        path.write_text(original)

    if proc.returncode != 0:
        first = next((l for l in proc.stdout.splitlines() if l.startswith("FAILED")), "")
        # A crash is not a catch. NameError/TypeError/AttributeError/SyntaxError almost always
        # mean the *mutation* was malformed -- an unimported symbol, a shape that no longer
        # lines up -- so the suite's real defence went untested and a genuine gap can hide
        # underneath. Twice already: sobol/undo-b1-blocks was concealing a test that
        # reimplemented its own subject. Flagged rather than counted silently.
        if any(e in first for e in ("NameError", "SyntaxError", "AttributeError", "TypeError")):
            return True, f"SUSPECT (crash, not assertion): {first[:70]}"
        return True, first[:88] or "caught"
    return False, "SURVIVED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="substring filter on the mutation id")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args(argv)

    chosen = [m for m in MUTATIONS if args.k in m.id]
    print(f"{len(chosen)} mutations\n")
    survivors, suspect = [], []
    for m in chosen:
        t = time.time()
        caught, detail = run(m, args.timeout)
        mark = ("SUSPECT " if detail.startswith("SUSPECT") else "caught ") if caught else "SURVIVED"
        print(f"  [{mark}] {m.id:<28} {time.time() - t:5.1f}s  {detail}")
        if not caught:
            survivors.append((m, detail))
        elif detail.startswith("SUSPECT"):
            suspect.append((m, detail))

    print()
    if suspect:
        print(f"{len(suspect)} mutation(s) caught by a CRASH rather than an assertion. Each is "
              "probably malformed;\nfix it and re-run, because a crash proves the harness ran "
              "and nothing more:\n")
        for m, detail in suspect:
            print(f"  {m.id}\n    {detail}\n")
    if not survivors:
        print(f"all {len(chosen)} mutations caught"
              + (f", but {len(suspect)} by crashing." if suspect else "."))
        return 1 if suspect else 0
    print(f"{len(survivors)} SURVIVOR(S) — each is a real gap:\n")
    for m, detail in survivors:
        print(f"  {m.id}\n    {m.path}: {m.why}\n    {detail}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
