"""Type aliases, in one place so annotations elsewhere stay short.

docs/conventions.md: public functions are annotated, and docstrings state shapes.
`A: (n_members, m, r)` is worth more than a paragraph of prose.
"""

from typing import Any, TypeAlias

import jax

Array: TypeAlias = jax.Array

# JAX has no distinct static type for a typed PRNG key: `jax.random.key(0)` is a jax.Array
# with an extended dtype. The alias exists to say which one a function wants.
Key: TypeAlias = jax.Array

# No structural type for pytrees either. Any is honest; the shape contract lives in the
# docstring.
PyTree: TypeAlias = Any
