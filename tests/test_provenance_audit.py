"""`experiments/provenance_audit.py`: every commit a record cites must stay fetchable.

Loaded by path, like the other drivers, because experiments/ is not a package. This file
moves with the experiments when the repository splits (docs/14).
"""

import importlib.util
import json
import pathlib

import pytest

SCRIPT = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
          / "provenance_audit.py")
spec = importlib.util.spec_from_file_location("provenance_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

A, B, C = "a" * 40, "b" * 40, "c" * 40


def test_a_stamp_is_owned_by_where_it_sits():
    doc = {"env": {"commit": A, "hyperscalees": {"commit": B}},
           "rows": [{"env": {"commit": A}}, {"env": {"commit": C, "shardes": {"commit": B}}}]}
    found = sorted(audit.citations(doc), key=str)
    assert (None, A, False) in found and found.count((None, A, False)) == 2
    assert ("hyperscalees", B, False) in found
    assert (None, C, True) in found, "an env carrying a shardes block is a post-split record"
    assert ("shardes", B, False) in found


def test_only_full_hex_shas_count():
    doc = {"env": {"commit": "unknown"}, "other": {"commit": "abc123"}, "n": {"commit": 7}}
    assert list(audit.citations(doc)) == []


def test_after_the_split_a_record_without_a_library_block_is_legacy(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"env": {"commit": A}}))
    new = tmp_path / "new.jsonl"
    new.write_text(json.dumps({"env": {"commit": C, "shardes": {"commit": B}}}) + "\n"
                   + "not json, a killed writer left this\n")

    together = audit.collect([old, new], split=False)
    assert set(together[audit.SELF]) == {A, C}, "one tree: every own stamp is this repo's"

    apart = audit.collect([old, new], split=True)
    assert set(apart[audit.LEGACY]) == {A}
    assert set(apart[audit.SELF]) == {C}
    assert set(apart["shardes"]) == {B}


def test_every_commit_this_repo_cites_is_reachable():
    """The real thing, on the real records. Fails the day a rebase orphans a cited commit."""
    root = audit.ROOT
    if audit.git(root, "rev-parse", "--is-inside-work-tree") != "true":
        pytest.skip("not a git checkout")
    if audit.git(root, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: reachability cannot be decided without full history")
    trunk = "origin/main" if audit.git(root, "rev-parse", "--verify", "-q", "origin/main") else "main"
    report = audit.audit(root, trunk=trunk)
    assert report["owners"][audit.SELF]["distinct"] >= 46
    assert report["unreachable"] == [], (
        "no ref reaches these cited commits; tag them (git tag provenance/<sha> <sha>) and "
        f"push the tag: {report['unreachable']}")


def test_an_orphaned_commit_fails_the_audit(tmp_path):
    """A guard that cannot fail is not a guard. Build a repo whose one record cites a
    commit that exists nowhere, and the same record citing HEAD, and check both verdicts."""
    import subprocess

    def run(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    record = tmp_path / "experiments" / "results" / "cell.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"env": {"commit": "d" * 40}}))
    run("add", "-A")
    run("commit", "-q", "-m", "a record citing a commit nobody has")

    report = audit.audit(tmp_path, trunk="main")
    assert report["unreachable"] == [(audit.SELF, "d" * 40, 1)]

    head = audit.git(tmp_path, "rev-parse", "HEAD")
    record.write_text(json.dumps({"env": {"commit": head}}))
    run("commit", "-q", "-am", "now it cites a commit that exists")
    report = audit.audit(tmp_path, trunk="main")
    assert report["unreachable"] == []
    assert report["owners"][audit.SELF]["off_trunk"] == {}, "HEAD~1 is on main"
