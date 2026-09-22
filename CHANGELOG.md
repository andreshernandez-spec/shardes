# Changelog

The supported surface is `shardes.__all__`. Before 1.0 a minor version may change it, and
says so here. Everything else is importable and may move without notice.

## 0.1.1

Packaging only; the code is 0.1.0's.

- First release on PyPI: `pip install shardes`.
- Index metadata (classifiers, keywords, links), and an explicit list of what the sdist
  contains.
- A release workflow: it builds from the tag, checks that the tag names the declared
  version, checks the artifacts, installs the wheel into a clean environment, and only
  then publishes, through PyPI's trusted publishing rather than a token.

## 0.1.0

The first version with a repository of its own. Until the tag `monorepo-final` the library
shared one with the paper it was built for, which is now
[shardes-paper](https://github.com/andreshernandez-spec/shardes-paper).

- `ShardedES`: `init`, `ask`, `apply`, `tell`, over the one-axis device mesh `make_mesh`
  builds.
- Strategies: `IIDGaussian`, `SeedRegenerated(chunk)`, `LowRank(r)`, and `Mirrored(inner)`
  for antithetic pairs around any of them.
- Two placements of the update contraction, `how="A"` and `how="B"`.
- Shaping: `centered_ranks` (the default), `centered`, `group_relative`, `none`.
- The seams a model routes its weights through, `shardes.nn.dense` and `shardes.nn.embed`,
  and `shardes.check.check_model` to find the weights that miss them.
- Problems: `quadratic`, `mlp`, `transformer_block`, and a Qwen2.5 port that needs
  `shardes[models]`.
- Python 3.12 or newer, JAX 0.11 or newer.
