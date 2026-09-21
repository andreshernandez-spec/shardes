# 14 - Splitting the repository: a library, and the paper that uses it

Status: plan, 2026-09-21, merged in #122. Phase 0 is done and phase 1 is under way as
separate PRs. Four corrections made while executing it are marked "Corrected". Every fact in the first section
was measured on main at f645e00 and the commands are given so it can be re-measured.

## Why

`shardes` calls itself a library and is laid out as a research project with a library
inside it. 2,970 of its 3,082 tracked files are experiments. The README is a phase and
gate status page, and a stale one: it still says no comparison against an external
reference has been run. `pip install` gives version 0.0.0 with nothing exported from the
top level. There is no CI. A third of the test suite tests experiment drivers, loaded by
file path, not the library.

The paper frames the library as "the instrument, cited and released". An instrument
someone can install in one line, read the README of, and use without opening a paper is
a stronger citation than a directory in a research repo. The split is what makes that
true.

## What the plan rests on

| # | Fact | How measured |
|---|---|---|
| 1 | The wheel builds, and every submodule imports from a plain install outside the repo. Missing: `__version__`, any top-level export, `py.typed`. Version is 0.0.0. | `pip wheel . --no-deps`, install with `--target`, import from `/tmp` |
| 2 | Hard dependencies are jax, numpy, scipy. `qwen2.load` imports `safetensors`, which no extra declares. `problems/control.py` imports `mujoco_playground`, declared under `tasks`. | grep of `src/` imports |
| 3 | The GitHub-side repo is 12.5 MB. History is cheap to keep. The local `.git` is 1.7 GiB because of one unreachable 2 GB blob, a checkpoint that was once staged; it was never pushed. | `gh api repos/.../shardes --jq .size`; `git verify-pack` |
| 4 | 2,727 tracked result files carry 2,823 commit stamps over 46 distinct shardes SHAs. 45 are ancestors of main. One, `1ba0dd0` (34 records: the v5e contraction and regen cells, 10 E17b cells), was orphaned by the rebase in #116 and was reachable from no ref. Pinned on 2026-09-21 by the tag `provenance/1ba0dd0`, the one step of this plan already taken. | scan of `"commit"` fields, `git merge-base --is-ancestor` |
| 5 | A 47th SHA, `b77f7d6`, is the EGGROLL reference's commit, stamped under `env/hyperscalees/commit`. The harness already records a second repository's commit. | `results-m4-tpu-v5e8` |
| 6 | 20 places hardcode the repo URL. Committed Kaggle kernels and pod scripts clone it at a pinned SHA and `pip install -e` it. | `git grep github.com/andreshernandez-spec/shardes` |
| 7 | 271 test functions exercise the library. 126, in five files, exercise experiment drivers: `test_phase0_driver`, `test_phase2_driver`, `test_m4_eggroll`, `test_countdown_task`, `test_accelerator_coverage`. | grep for `experiments` paths in `tests/` |
| 8 | 90 tracked experiment scripts. Eight put `src/` on `sys.path` by hand, the rest rely on the editable install. They use `core`, `strategies.*`, `problems`, `contraction`, `sharding`, `shaping`, `dimensions`, `estimator`, `coupling`, and two private knobs: `lowrank.PAD_RANK1` and `sharding.POP`. | `git ls-files` plus grep |
| 9 | `harness.capture_env` stamps one commit and one dirty flag. Today that covers `src/` because it is the same tree. | `experiments/harness.py` |
| 10 | State git will not move: `drafts/` (local by the tree rule), 3.4 GB of smoke checkpoints, ignored logs under `experiments/`. | `git status --ignored` |
| 11 | Free names: `shardes` on PyPI, `shardes-paper` on GitHub. `git filter-repo` is installed. | pypi.org JSON endpoint, `gh api` |

## The shape afterwards

**`andreshernandez-spec/shardes`**: the existing repository, same URL, history kept.

```
src/shardes/     unchanged, plus a public surface and __version__
tests/           the library's tests, and tests/gpu
validation/      real-hardware checks of the library (today experiments/phase1:
                 reference.py, reference.json, memory.py, the kernels that run the
                 suite on 2x T4)
tools/           mutation.py, lowrank_compile_diag.py, the bf16 probe, probe_lr1.py
examples/        quickstart, a sharded run on simulated devices, porting a model
                 through shardes.nn
docs/            design, conventions, diagnoses, proposals
README.md        written for a user
CHANGELOG.md, CITATION.cff, .github/workflows/
```

**`andreshernandez-spec/shardes-paper`**: new, carrying the history of its own paths.

```
paper/             unchanged
experiments/       phase0, phase2, countdown, harness.py, kernels, results, figures
tests/             the five driver test files
docs/              campaign docs, runbook, preregistrations; PLAN.md at the root
requirements.lock  pins shardes to a tag or SHA, and everything else
pyproject.toml     not a package to publish: the pinned dependency and the extras
experiments/provenance_audit.py
```

The paper repo keeps today's directory names, so the Makefile, `\graphicspath`, and every
relative path inside the drivers work unchanged.

## The decision that settles most of the others: keep the library repo's history

| Option | What happens to the 46 stamped SHAs and the pinned kernels | Verdict |
|---|---|---|
| A. Keep `shardes` history. Delete the moved paths from HEAD in one ordinary commit. Build the paper repo with `git filter-repo`. | All resolve at the URL they were recorded against. A checkout of any old SHA is the whole monorepo as it was, experiments included. | **Recommended** |
| B. Rewrite `shardes` to `src/` and `tests/` only. | Every stamp, README citation and pinned kernel breaks unless a third archive repo is kept forever. Saves about 10 MB. | Reject |
| C. Rename the existing repo to the paper repo and create a new `shardes`. | GitHub's redirect dies when the old name is reused, so old kernel URLs land in a repo where their SHAs do not exist. | Reject |

Option A costs a 12.5 MB clone and buys a clean provenance story: **the library repo's
history is the archive of the monorepo era.** A record stamped before the split cites
`shardes@<sha>`; check that SHA out and the driver, the config and the library are all
there, as run. The paper repo's README says so in its first screen.

The paper repo's history is rewritten by the filter, so its SHAs are new. Nothing cites
them yet, so nothing breaks.

## Provenance once there are two repos

One pin stays the source of truth: **the paper repo's SHA determines everything**, because
its lock file pins the library. A kernel or a pod clones the paper repo at a SHA and
installs from the lock. Nobody types a library SHA by hand.

`capture_env` grows a block next to the fields it has, on the pattern it already uses
for the EGGROLL reference:

```json
"commit": "<paper repo HEAD>",
"dirty_worktree": false,
"shardes": {"version": "0.1.0", "commit": "<library SHA>", "dirty": false,
            "source": "vcs"}
```

`commit` comes from `direct_url.json` (PEP 610) for a git install. For an editable
install it comes from running git in the directory the install points at, which is also
where `dirty` comes from. A wheel from an index has a version and no commit, and
`source` says so.

Three rules go with it:

- A campaign driver refuses a dirty or unpushed library, the way `e13_campaign.sh`
  already refuses a dirty tree. A probe may run against one and the record says so.
- A record with no `shardes` block is from the monorepo era, and its library commit is
  its `commit`. Readers treat the missing key that way and nothing is backfilled.
- `experiments/provenance_audit.py` extracts every stamped SHA from every record and
  checks that a ref in the named repository reaches it. It runs in the paper repo's CI.
  It would have caught `1ba0dd0` the day #116 was rebased. Corrected: this said
  `tools/`, but `tools/` in the monorepo goes to the library and the audit belongs with
  the records, so it lives where the filter will carry it.

## File mapping

| Today | Goes to | Note |
|---|---|---|
| `src/`, `conftest.py`, `LICENSE` | library | `conftest.py` is the worktree import fix and belongs with the suite |
| `tests/` except the five driver files | library | |
| `tests/test_phase0_driver.py`, `test_phase2_driver.py`, `test_m4_eggroll.py`, `test_countdown_task.py` | paper | they load drivers by path |
| `tests/test_accelerator_coverage.py` | split | the `reference.json` half stays with the library; the half that reads phase 2 sweep configs goes with them |
| `tests/gpu/`, `experiments/phase1/reference.py`, `reference.json`, `memory.py`, `phase1/kaggle/` | library, under `validation/` | gate G1's real-hardware check of the library, not a paper result |
| `experiments/mutation.py`, `lowrank_compile_diag.py`, `bf16/probe.py`, `phase2/probe_lr1.py` | library, under `tools/` | each backs a library decision: the suite, compile cost, the fitness dtype guard, the rank-1 pad |
| `experiments/phase1/comms.py`, `sobol_b1.py` | paper | `comms.py` produces byte counts the paper quotes; `sobol_b1.py` is a research question |
| `experiments/phase2/probe_eggroll_tpu.py` | paper | it is about the reference implementation |
| `experiments/harness.py`, `phase0/`, `phase2/`, `countdown/` | paper | configs, drivers, kernels, results, figures |
| `paper/` | paper | |
| `docs/00`, `01`, `02`, `04`, `compute`, `conventions`, `diagnosis-*`, `postmortem-*`, `proposal-*`, `BACKLOG` | library | library code cites these: 01 from 14 source files, 02 from 8 |
| `docs/03`, `05` to `13`, `PLAN.md` | paper | cited from experiments, not from `src/` (03: 24 to 1) |
| `docs/14-repo-split.md` | library | the record of the split, next to the history it explains |
| `pyproject.toml` | library keeps it; the paper repo gets a new one | the `experiments` extra moves with the experiments |
| `README.md`, `CLAUDE.md` | both get their own | see below |

A doc lives on the side whose code cites it more, and the other side links to it at a
tag. `BACKLOG` has items for both and is divided by item.

## Steps

Each phase ends in a state where everything works, and nothing is irreversible until
phase 3, which is itself an ordinary commit.

### Phase 0: freeze and protect, in the monorepo

1. ~~Tag main `monorepo-final`.~~ Corrected: not yet. Phase 1 keeps changing the
   monorepo, so a tag placed now would not be final. It goes on the parent of phase 3's
   removal commit, which really is the last commit where the whole project is one tree.
   Main's history is kept either way, so nothing needs protecting in the meantime.
2. Tag `provenance/1ba0dd0` and push it, so the orphaned commit behind 34 records can
   never be collected. **Done 2026-09-21**, ahead of the rest: the only things keeping
   that commit alive were a local reflog with about nine days left on it and GitHub's
   retention of unreachable objects, and neither is a promise.
3. Add `experiments/provenance_audit.py` and run it. **Done**: 2,727 files, 46 own
   commits in 2,781 stamps, all reachable, `1ba0dd0` reported as kept by its tag, and one
   foreign commit under `hyperscalees` listed as unchecked. Its tests include a repo
   built to fail it.
4. Optional and local: `git gc --prune=now` in `es/` drops the 2 GB blob.

### Phase 1: make the library stand alone while still inside the monorepo

Five small PRs to main. The experiments keep running throughout, which is the point of
doing this before moving anything.

1. **Public surface.** `shardes/__init__.py` exports `ShardedES`, `IIDGaussian`,
   `SeedRegenerated`, `LowRank`, the `Mirrored` wrapper, `make_mesh`, and `__version__`. Deep imports keep working, since the
   paper's 90 scripts use them. Version becomes `0.1.0.dev0`, single-sourced. Add
   `py.typed`.
2. **Extras.** Add `models = ["safetensors"]`. Corrected: this listed
   `huggingface_hub` too, but nothing in `src/` imports it; the experiments do, and
   their extra moves with them. Leave `experiments` for now.
3. ~~**No path hacks.**~~ Corrected: moved to phase 2. In a worktree those eight
   `sys.path.insert(... "src")` lines are what makes a driver import its own checkout
   instead of whichever one the editable install points at, the trap `conftest.py`
   documents. Deleting them while the library is still next door trades a visible hack
   for a silent wrong import. In the paper repo there is no `src/` for them to find, so
   they go in its first commit, where the verification gate proves the drivers run
   against the installed library.
4. **Two-sided provenance.** Extend `capture_env` as above. While both halves share a
   tree the two commits must be equal, and a test asserts it.
5. **Path-clean moves.** `experiments/phase1` to `validation/`, the four tools to
   `tools/`, and the split of `test_accelerator_coverage`. Doing the moves here keeps
   phase 2's filter a plain list of paths.
6. **CI.** A GitHub Actions workflow: `pytest --fast` on CPU with eight simulated
   devices on every PR, the full tier on main, and a job that builds the wheel,
   installs it into a clean environment and imports every submodule. The last job is
   what catches an undeclared dependency like `safetensors`.

### Phase 2: create the paper repo, touching nothing in the monorepo

1. Fresh clone, then `git filter-repo` keeping `paper/`, `experiments/`, the paper-side
   docs, `PLAN.md` and the driver tests, per the mapping table. First commit on top:
   delete the eight `sys.path.insert(... "src")` lines (see phase 1 step 3).
2. Add to the new repo: `pyproject.toml` with `shardes @ git+https://...@<SHA>` and
   extras for plotting, the HF stack and GRPO; `requirements.lock`; a README that opens
   with what the repo is and the monorepo-era provenance rule; `CLAUDE.md`; a test
   config; CI that runs the driver tests and the provenance audit.
3. **Verification gate.** In a clean venv built from the lock, all of these must hold:
   - the 126 driver tests pass;
   - `make -C paper tables` reproduces `paper/generated/*.tex` byte for byte;
   - every figure script runs, and the figures match by hash or by eye;
   - `latexmk` builds the same page count with no undefined references;
   - one smoke driver writes a record carrying both provenance blocks.
4. Update the current kernels and pod bootstraps to clone the paper repo at a SHA and
   install from the lock. Prove it with one free Kaggle T4 kernel before any TPU
   session depends on it. Old kernels stay as they are, in history, against the old
   URL, where they still work.
5. Create the GitHub repo, push, and turn on branch protection. This is the first
   outward-facing step and waits for a yes.

### Phase 3: remove what moved from the library repo

1. Tag main `monorepo-final`, annotated, and push it: the parent of what follows.
   Then one PR: `git rm -r experiments paper` and the paper-side docs, tests and
   `PLAN.md`.
   Rewrite the README for a user. Trim `CLAUDE.md` to the library's scope. Drop the
   `experiments` extra. Fix any docstring in `src/` that cites a doc that moved; a small
   test that every `docs/*.md` path named in `src/` exists keeps it fixed.
2. Tag the merge `v0.1.0`. Re-point the paper repo's pin from that SHA to the tag, which
   is the same commit.
3. In the paper: two URLs instead of one, the provenance paragraph restated (records
   before the split cite `shardes@sha`, records after cite the paper repo and a library
   version), and the reproduction appendix.

### Phase 4: the library as a product

Independent PRs; none blocks the paper.

- `examples/`: a quadratic in twenty lines, the same run sharded over eight simulated
  devices, and a model ported through `shardes.nn`. CI runs them.
- README: what it is, install, quickstart, choosing a strategy and a contraction
  placement (the paper's decision rule in five lines, with a citation), what the
  invariants guarantee, non-goals.
- `CHANGELOG.md`, `CITATION.cff`.
- A job in the paper repo that runs its driver tests against library main, allowed to
  fail. It reports drift early without forcing the pin to move.
- PyPI, if wanted (decision D4).

### Phase 5: this machine

- `es/` stays the library clone. `shardes-paper/` is cloned next to it.
- Move by hand what git will not: `drafts/` to the paper clone and into its
  `.git/info/exclude`; the ignored logs under `experiments/`; delete or move the 3.4 GB
  of smoke checkpoints.
- Recreate the venvs. Prune worktrees. Update the Kaggle and RunPod skill notes, which
  name `es/experiments/...` paths.

## What each repo tells a reader

**Library README**: one paragraph on what it is, install, quickstart, the two published
algorithms one argument apart (the existing example is good), a table of strategies and
placements, the invariants stated as guarantees, non-goals, how to cite. Phases, gates
and sweep history leave; they stay reachable at `monorepo-final`.

**Paper repo README**: what the paper claims, how to rebuild the PDF, how to rerun one
experiment end to end, the provenance rules, and where the results directories' own
READMEs are.

**`CLAUDE.md`**: the library keeps the invariants, the environment section, the test
tiers and the working style. The paper repo takes rule 2 (every number has a script),
the config-before-run rule, the runbook pointers and the pin policy. The identity and
no-upstream-push rules are in the tree-level file and apply to both.

## Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| D1 | Name and visibility of the paper repo | `shardes-paper`, public. The monorepo history is already public with the paper in it, so private hides nothing. If the venue is double-blind, the artifact is anonymized at submission whatever this repo is called. |
| D2 | Where library QA scripts live | `tools/` and `validation/` in the library, as mapped |
| D3 | Does `shardes.problems` stay in the library | Yes. It is 700 lines, the library's own tests need a model, and the Qwen port is the worked example of the `shardes.nn` seams. Rename nothing now. |
| D4 | PyPI | Reserve the name with `0.1.0` once the README is a user's README. A git pin is enough for the paper either way. |
| D5 | Rename the local `es/` to `shardes/` | Yes, at phase 5, with the skill notes |
| D6 | Paper repo history: filtered, or one import commit | Filtered. Blame on the paper and the drivers is worth keeping, and it costs nothing. |

## Risks

| Risk | Mitigation |
|---|---|
| The experiments lean on private API (`PAD_RANK1`, `POP`, deep imports), so a library refactor breaks them silently | The pin. The allowed-to-fail job against library main. Promote the two knobs to documented names in phase 1. |
| The Kaggle or pod bootstrap breaks and it is found during a TPU session that took hours to queue | Phase 2 step 4 proves the bootstrap on a free T4 first |
| The paper's text promises one repository | Phase 3 step 3, and the verification gate rebuilds the PDF |
| A record is stamped against a library commit nobody can fetch | The campaign drivers refuse; the audit runs in CI |
| The five driver tests silently stop being run by anyone | They are the paper repo's CI |
| Local state is lost in the move | Phase 5 is a checklist, and nothing in `es/` is deleted until the paper clone is verified |
| The two `CLAUDE.md` files drift | Shared rules stay in the tree-level file only |

## What this plan does not do

It does not rewrite any published history. It does not use a submodule: the lock file
gives the same exact pin, and a submodule makes every kernel bootstrap need
`--recurse-submodules`. It does not redesign the API: the public surface re-exports what
exists. It does not move results out of git: they are 17 MB.

## Effort

Phase 0 is one PR and an hour. Phase 1 is five small PRs, about a day. Phase 2 is half a
day plus one free Kaggle run. Phase 3 is one PR on each side. Phases 4 and 5 follow at
leisure. No paid compute anywhere.
