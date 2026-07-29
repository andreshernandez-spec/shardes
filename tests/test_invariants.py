"""The invariants from CLAUDE.md. Breaking one of these is a bug, not a tradeoff.

Implemented here:
    test_no_ravel_pytree          static check over src/

Waiting on the strategies (docs/01 C0.1):
    test_seed_by_member_index     member i's perturbation is identical regardless of n,
                                  batching, or device count
    test_lowrank_never_materializes
                                  trace the jaxpr under LowRank, assert no array of shape
                                  (n, m, k) with m, k > r appears
    test_f32_accumulation         reduction over 2^18 bf16 members accumulates in f32
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "shardes"

BANNED = "ravel_pytree"


def _python_files():
    files = sorted(SRC.rglob("*.py"))
    # Without this, a wrong SRC path makes the scan vacuous and the test passes by
    # finding nothing at all.
    assert files, f"no python files under {SRC}, the invariant test is not scanning src/"
    return files


def test_src_is_scannable():
    """Guards the guard. Named separately so a path break is not read as compliance."""
    assert len(_python_files()) >= 5


def test_no_ravel_pytree():
    """No global flattening anywhere under src/. Invariant 1 in CLAUDE.md.

    This is the architectural difference from evosax and the reason low-rank perturbation
    is expressible at all: `evosax/algorithms/base.py` sets `num_dims` from
    `ravel_pytree(...)`, which forecloses per-matrix structure.

    Checked with ast rather than a text search, because several modules mention
    ravel_pytree in their docstrings precisely to say they must not call it. A grep would
    fire on the prose and a maintainer would learn to ignore this test.
    """
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            hit = (
                (isinstance(node, ast.Name) and node.id == BANNED)
                or (isinstance(node, ast.Attribute) and node.attr == BANNED)
                or (
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    and any(a.name == BANNED or a.asname == BANNED for a in node.names)
                )
            )
            if hit:
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")

    assert not offenders, f"{BANNED} used under src/: {offenders}"
