"""Code and comments here must not name a local path that is not here.

The experiments, the paper and their docs moved to the shardes-paper repository
(docs/14). Comments in this one still cite them as evidence, and should: but as
`shardes-paper:experiments/...`, so nobody goes looking for a directory that is not in
this checkout. And a `docs/` file named from the code has to exist.

Only code-adjacent text is checked. The narrative in docs/ describes the project's
history, including where things used to be, and is left alone.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKED = [p for d in ("src", "tests", "tools", "validation", "examples")
           for p in (ROOT / d).rglob("*") if p.suffix in (".py", ".md") and p.name != pathlib.Path(__file__).name]
# The two files a newcomer reads first.
CHECKED += [ROOT / "README.md", ROOT / "CLAUDE.md"]
#: What lives in the other repository. `shardes-paper:` in front of it is the convention.
ELSEWHERE = re.compile(r"(?<![\w:-])(?:experiments|paper)/[\w./-]+|(?<![\w:-])PLAN\.md")
DOC = re.compile(r"(?<![\w:-])docs/([\w.-]+?\.md|\d\d)(?![\w-])")
#: A docs/ path that is somebody else's, quoted. CLAUDE.md cites JAX's contributing guide.
NOT_OURS = {("CLAUDE.md", "contributing.md")}


def test_nothing_names_a_directory_that_moved_without_saying_where():
    offenders = [f"{p.relative_to(ROOT)}: {m.group(0)}"
                 for p in CHECKED for m in ELSEWHERE.finditer(p.read_text())]
    assert not offenders, (
        "these live in shardes-paper now; write shardes-paper:<path>:\n  " + "\n  ".join(offenders))


def test_every_doc_named_from_the_code_exists():
    have = {p.name for p in (ROOT / "docs").glob("*.md")}
    numbered = {name[:2] for name in have if name[:2].isdigit()}
    missing = []
    for p in CHECKED:
        for m in DOC.finditer(p.read_text()):
            ref = m.group(1)
            if (str(p.relative_to(ROOT)), ref) in NOT_OURS:
                continue
            if (ref.isdigit() and ref not in numbered) or (not ref.isdigit() and ref not in have):
                missing.append(f"{p.relative_to(ROOT)}: docs/{ref}")
    assert not missing, (
        "no such file in docs/; if it moved to the paper repository, write "
        "shardes-paper:docs/<name>:\n  " + "\n  ".join(sorted(set(missing))))
