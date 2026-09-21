"""`harness.library_provenance`: a record has to say which shardes did the arithmetic.

Loaded by path, like the other drivers, because experiments/ is not a package. This file
moves with the experiments when the repository splits (docs/14).
"""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("harness", ROOT / "experiments" / "harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

SHA = "c" * 40


class FakeDist:
    version = "9.9.9"

    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return self._direct_url


def not_a_checkout(here, *args):
    return ""  # what `git` prints for an untracked path or outside any repository


def test_in_one_tree_the_library_commit_is_the_drivers_commit():
    """The monorepo invariant, and the audit relies on it: a record whose two commits are
    equal was written while the library was a directory of the same tree."""
    env = harness.capture_env(ROOT / "experiments" / "phase2", ())
    lib = env["shardes"]
    assert lib["source"] == "checkout", lib
    assert lib["commit"] == env["commit"] and len(lib["commit"]) == 40
    assert isinstance(lib["dirty"], bool) and isinstance(lib["version"], str)


def test_the_old_fields_are_untouched():
    env = harness.capture_env(ROOT / "experiments" / "phase2", ())
    for key in ("commit", "dirty_worktree", "jax", "jaxlib", "numpy", "scipy", "python",
                "platform", "hostname", "device_count", "device_kind", "device_platform",
                "xla_flags", "jax_platforms"):
        assert key in env, key


def test_an_edited_library_is_dirty_and_an_unanswerable_git_is_too():
    def edited(here, *args):
        if args[0] == "ls-files":
            return "__init__.py"
        if args[0] == "status":
            return " M core.py"
        return SHA

    assert harness.library_provenance(edited) == {
        "version": harness.library_provenance(edited)["version"],
        "commit": SHA, "dirty": True, "source": "checkout"}

    def git_is_broken(here, *args):
        return "__init__.py" if args[0] == "ls-files" else "unknown"

    assert harness.library_provenance(git_is_broken)["dirty"] is True


@pytest.mark.parametrize("direct_url, expected", [
    (json.dumps({"url": "https://github.com/x/shardes", "vcs_info": {"vcs": "git",
                                                                     "commit_id": SHA}}),
     {"commit": SHA, "dirty": False, "source": "vcs"}),
    (json.dumps({"url": "file:///somewhere/shardes", "dir_info": {}}),
     {"commit": "unknown", "dirty": True, "source": "local"}),
    (None, {"commit": "unknown", "dirty": False, "source": "index"}),
], ids=["git-install", "local-directory", "package-index"])
def test_outside_a_checkout_pip_s_record_decides(monkeypatch, direct_url, expected):
    monkeypatch.setattr(harness.metadata, "distribution", lambda name: FakeDist(direct_url))
    got = harness.library_provenance(not_a_checkout)
    assert {k: got[k] for k in expected} == expected


def test_a_venv_inside_the_repo_is_not_the_checkout(monkeypatch):
    """site-packages under an in-repo .venv sits inside the work tree and is tracked by
    nothing. `rev-parse --show-toplevel` would say yes; `ls-files --error-unmatch` says no."""
    calls = []

    def git_in(here, *args):
        calls.append(args[0])
        return "" if args[0] == "ls-files" else "/the/repo"

    monkeypatch.setattr(harness.metadata, "distribution", lambda name: FakeDist(None))
    assert harness.library_provenance(git_in)["source"] == "index"
    assert calls == ["ls-files"], "asked whether the file is tracked, and stopped there"
