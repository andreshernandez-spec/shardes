"""shardes: sharded evolution strategies for JAX.

An ask / evaluate / tell core that shards the population across devices, with the
perturbation scheme as a pluggable strategy. The two published LLM-scale algorithms are
one argument apart:

    import shardes

    mesh = shardes.make_mesh()                       # every visible device, one "pop" axis

    # Qiu et al. 2025: full-rank noise regenerated from seeds, small population
    es = shardes.ShardedES(shardes.Mirrored(shardes.SeedRegenerated()),
                           n=32, sigma=1e-3, lr=5e-7, mesh=mesh)
    # EGGROLL (Sarkar et al. 2025): rank-1 factors, never materialized, huge population
    es = shardes.ShardedES(shardes.Mirrored(shardes.LowRank(r=1)),
                           n=262_144, sigma=1e-3, lr=5e-7, mesh=mesh)

    state = es.init(key, params)
    pert, state = es.ask(state)                      # a Perturbation, never a batch of params
    fitness = es.apply(model, state, pert)(batch)    # one fitness per member
    state = es.tell(state, pert, fitness)

`tests/test_public_api.py` runs exactly this, so it cannot rot.

The names below are the supported surface. Everything else stays importable from its
module (`shardes.nn` for the seams a model routes its matmuls through, `shardes.shaping`,
`shardes.sharding`, `shardes.contraction`, `shardes.coupling`), and the module docstrings
say what belongs where and which section of `docs/` specifies it.
"""

from shardes import contraction, coupling, nn, shaping, sharding
from shardes.core import ShardedES, State
from shardes.sharding import make_mesh
from shardes.strategies.iid_gaussian import IIDGaussian
from shardes.strategies.lowrank import LowRank
from shardes.strategies.mirrored import Mirrored
from shardes.strategies.seed_regenerated import SeedRegenerated

#: Single source of the version: pyproject.toml reads it from here (tool.hatch.version).
__version__ = "0.2.0.dev0"

__all__ = [
    "IIDGaussian",
    "LowRank",
    "Mirrored",
    "SeedRegenerated",
    "ShardedES",
    "State",
    "__version__",
    "contraction",
    "coupling",
    "make_mesh",
    "nn",
    "shaping",
    "sharding",
]
