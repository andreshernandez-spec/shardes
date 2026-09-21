"""The supported surface: what `import shardes` promises a user.

Everything in the library stays importable from its module, and the experiments use those
deep paths. This pins the short list a README can show: that each name exists, is the same
object as its deep import, and that the quickstart in `shardes/__init__.py` actually runs
using nothing but those names.
"""

import re

import jax
import jax.numpy as jnp
import pytest

import shardes
from shardes.problems import transformer_block


def test_every_public_name_resolves_and_the_list_is_tidy():
    assert sorted(set(shardes.__all__)) == shardes.__all__, "sorted, no duplicates"
    for name in shardes.__all__:
        assert hasattr(shardes, name), name


def test_the_short_names_are_the_deep_imports_not_copies():
    from shardes.core import ShardedES, State
    from shardes.sharding import make_mesh
    from shardes.strategies.iid_gaussian import IIDGaussian
    from shardes.strategies.lowrank import LowRank
    from shardes.strategies.mirrored import Mirrored
    from shardes.strategies.seed_regenerated import SeedRegenerated

    assert shardes.ShardedES is ShardedES
    assert shardes.State is State
    assert shardes.make_mesh is make_mesh
    assert shardes.IIDGaussian is IIDGaussian
    assert shardes.LowRank is LowRank
    assert shardes.Mirrored is Mirrored
    assert shardes.SeedRegenerated is SeedRegenerated


def test_the_version_is_one_a_package_index_would_accept():
    # PEP 440, the subset this project will ever use: N.N.N with an optional dev/rc tail.
    assert re.fullmatch(r"\d+\.\d+\.\d+((\.dev|rc)\d+)?", shardes.__version__), shardes.__version__


def test_pyproject_reads_the_version_from_the_package():
    """One source. A literal in pyproject.toml beside `__version__` is two facts that drift."""
    import pathlib
    import tomllib

    meta = tomllib.loads(
        (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "version" in meta["project"]["dynamic"]
    assert "version" not in meta["project"]
    assert meta["tool"]["hatch"]["version"]["path"] == "src/shardes/__init__.py"


@pytest.mark.parametrize("make_strategy", [
    lambda: shardes.Mirrored(shardes.SeedRegenerated()),
    lambda: shardes.Mirrored(shardes.LowRank(r=1)),
    lambda: shardes.IIDGaussian(),
], ids=["qiu", "eggroll", "iid"])
def test_the_quickstart_runs_with_public_names_only(make_strategy):
    key = jax.random.key(0)
    params = transformer_block.init(key, d_model=16)
    batch = transformer_block.make_batch(jax.random.fold_in(key, 1), d_model=16,
                                         batch=2, seq=4)
    mesh = shardes.make_mesh()
    n = 4 * mesh.size                      # a whole number of mirrored pairs per device
    es = shardes.ShardedES(make_strategy(), n=n, sigma=1e-2, lr=1e-2, mesh=mesh)

    state = es.init(key, params)

    @jax.jit
    def generation(state):
        pert, state = es.ask(state)
        fitness = es.apply(transformer_block.loss, state, pert)(batch)
        return es.tell(state, pert, fitness), fitness

    new, fitness = generation(state)
    assert fitness.shape == (n,)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(new.params))
    moved = sum(float(jnp.sum(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(new.params), jax.tree.leaves(params)))
    assert moved > 0.0, "tell returned the parameters it was given"
