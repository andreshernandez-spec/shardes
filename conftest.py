"""Make `pytest` test THIS checkout, not whichever one pip installed.

`shardes` is installed editable, and an editable install points at one directory. Run the
suite from a git worktree and every `import shardes` resolves to the OTHER checkout, so
the tests pass or fail on code that is not the code under test. That is not hypothetical:
on 2026-08-24 three tests failed in a worktree because the editable target was on a branch
without `PAD_RANK1`, and earlier runs in the same session reported green while testing a
different branch than the one being reviewed.

Prepending this file's `src/` fixes it for every invocation, with no flag to remember.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path[:1]:
    sys.path.insert(0, str(SRC))
