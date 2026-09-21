# docs

What stayed with the library when the experiments and the paper moved to
[shardes-paper](https://github.com/andreshernandez-spec/shardes-paper) (see
`14-repo-split.md`).

| file | what it is |
|---|---|
| `00-context.md` | the two papers, the gap, prior art |
| `01-phase0-estimator-harness.md` | the estimator study that set the API's requirements, and its answer on coupling |
| `02-phase1-sharded-core.md` | the design of the core, and gate G1's criteria |
| `04-phase3-coupling.md` | coupled sampling at scale: dropped, kept as the record of what was predicted |
| `14-repo-split.md` | how and why this repository was split, and the provenance rules that follow |
| `conventions.md`, `compute.md` | code, test and numerics conventions; development-time compute |
| `diagnosis-*`, `postmortem-*`, `proposal-*` | decisions and the evidence behind them |
| `BACKLOG.md` | deferred questions and what would settle each |

These files are the project's working record, and they describe things as they were when
written. A path beginning `experiments/` or `paper/`, `PLAN.md`, and docs 03 and 05 to 13
are in shardes-paper now. At the tag `monorepo-final` they are all in this repository.
