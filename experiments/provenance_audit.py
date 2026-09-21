#!/usr/bin/env python
"""Can every commit a result record cites still be fetched?

    python experiments/provenance_audit.py
    python experiments/provenance_audit.py --repo shardes=../shardes --legacy ../shardes

Every result record stamps the commit it was measured at (`harness.capture_env`), and the
paper promises that each result traces to one. That promise is only as good as the
commit's reachability: a SHA that no branch or tag reaches is kept by the host at its
discretion and pruned locally after a reflog expires.

It has already happened once. Rebasing a branch for #116 orphaned `1ba0dd0`, which 34
records cite, and nothing noticed for three weeks. It is pinned now by the tag
`provenance/1ba0dd0`. This script is the thing that would have noticed.

What counts as a citation: any key named `commit` holding a 40-character hex string, in
any tracked `.json` or `.jsonl` under `experiments/`. Whose commit it is comes from where
the key sits:

    env.commit                      this repository
    env.<name>.commit               another repository, by name (`hyperscalees` today,
                                    `shardes` once the library lives in its own repo)

Commits in this repository are checked against every ref here. Another repository's are
checked only if `--repo NAME=PATH` points at a clone of it; otherwise they are listed as
unchecked rather than guessed at.

`--legacy PATH` is for after the split (docs/14). A record with no `env.shardes` block was
written when the library and the experiments were one tree, so its `env.commit` is a
commit of that tree, and that history stays in the library repository. Pointing `--legacy`
at a clone of it checks those records there instead of here.

Exit status is 1 if any checked commit is unreachable, 2 if the clone is shallow and the
question cannot be answered, 0 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SHA = re.compile(r"[0-9a-f]{40}")
SELF = "(this repository)"
LEGACY = "(monorepo era, this tree's history)"
#: The block `capture_env` adds once the library has its own repository. Its presence is
#: what marks a record as written after the split.
LIBRARY_BLOCK = "shardes"


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=False).stdout.strip()


def citations(node, parent: str | None = None):
    """Yield (owner, sha, has_library_block) for every commit stamp under `node`.

    `owner` is None for this repository's own stamp and the enclosing key for a foreign
    one. `has_library_block` is reported with an own stamp, since it decides whether that
    stamp belongs to the monorepo era.
    """
    if isinstance(node, dict):
        value = node.get("commit")
        if isinstance(value, str) and SHA.fullmatch(value):
            foreign = parent not in (None, "env")
            yield (parent if foreign else None, value, LIBRARY_BLOCK in node)
        for key, child in node.items():
            yield from citations(child, key)
    elif isinstance(node, list):
        for child in node:
            yield from citations(child, parent)


def documents(path: pathlib.Path):
    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    try:
        yield json.loads(text)
    except json.JSONDecodeError:
        return


def collect(files, split: bool) -> dict:
    """owner -> Counter of commits. `split` says the library has its own repository, which
    is what makes a record without a library block a monorepo-era one."""
    stamps: dict = collections.defaultdict(collections.Counter)
    for path in files:
        for doc in documents(path):
            for owner, sha, has_block in citations(doc):
                if owner is not None:
                    stamps[owner][sha] += 1
                elif split and not has_block:
                    stamps[LEGACY][sha] += 1
                else:
                    stamps[SELF][sha] += 1
    return stamps


def reachable(repo: pathlib.Path) -> tuple[set, bool]:
    shallow = git(repo, "rev-parse", "--is-shallow-repository") == "true"
    return set(git(repo, "rev-list", "--all").split()), shallow


def keepers(repo: pathlib.Path, sha: str, trunk: str) -> str:
    """Which refs reach `sha`: the trunk if it does, otherwise whatever else does."""
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, trunk],
                      capture_output=True, check=False).returncode == 0:
        return trunk
    refs = git(repo, "for-each-ref", "--contains", sha, "--format=%(refname:short)").split()
    return ", ".join(refs[:3]) + (" ..." if len(refs) > 3 else "")


def audit(root: pathlib.Path = ROOT, repos: dict | None = None,
          legacy: pathlib.Path | None = None, trunk: str = "origin/main") -> dict:
    repos = dict(repos or {})
    files = [root / f for f in git(root, "ls-files", "experiments").splitlines()
             if f.endswith((".json", ".jsonl"))]
    stamps = collect(files, split=legacy is not None)

    where = {SELF: root, LEGACY: legacy, **{k: pathlib.Path(v) for k, v in repos.items()}}
    report = {"files": len(files), "owners": {}, "unreachable": [], "shallow": False}
    for owner, shas in sorted(stamps.items()):
        repo = where.get(owner)
        entry = {"distinct": len(shas), "stamps": sum(shas.values()), "checked": repo is not None,
                 "off_trunk": {}}
        if repo is not None:
            known, shallow = reachable(repo)
            report["shallow"] |= shallow
            for sha, count in shas.items():
                if sha not in known:
                    report["unreachable"].append((owner, sha, count))
                else:
                    kept = keepers(repo, sha, trunk)
                    if kept != trunk:
                        entry["off_trunk"][sha] = (count, kept)
        report["owners"][owner] = entry
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", action="append", default=[], metavar="NAME=PATH",
                    help="a clone in which to check commits stamped under env.NAME")
    ap.add_argument("--legacy", type=pathlib.Path,
                    help="a clone holding the monorepo-era history (docs/14)")
    ap.add_argument("--trunk", default="origin/main")
    args = ap.parse_args(argv)

    report = audit(ROOT, dict(item.split("=", 1) for item in args.repo), args.legacy,
                   args.trunk)
    print(f"{report['files']} tracked result files under experiments/\n")
    for owner, e in report["owners"].items():
        state = "checked" if e["checked"] else "NOT CHECKED, no clone given"
        print(f"{owner}: {e['distinct']} distinct commits in {e['stamps']} stamps ({state})")
        for sha, (count, kept) in e["off_trunk"].items():
            print(f"    {sha[:10]}  {count:4d} stamps  not on {args.trunk}, kept by: {kept}")
    if report["shallow"]:
        print("\nSHALLOW CLONE: history is truncated, so reachability cannot be decided. "
              "Fetch with full depth and tags.")
        return 2
    if report["unreachable"]:
        print("\nUNREACHABLE, no ref reaches these. Tag them before they are pruned:")
        for owner, sha, count in report["unreachable"]:
            print(f"    {sha}  {count:4d} stamps  {owner}")
        return 1
    print("\nevery checked commit is reachable from a ref")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
