"""The README's quickstart is `examples/quickstart.py`, and this runs it.

In a subprocess, because it sets XLA_FLAGS before importing jax, which is the honest way to
run it and cannot be done from inside a process that already has.
"""

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_quickstart_runs_and_descends():
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src")}
    env.pop("XLA_FLAGS", None)  # the example sets it; conftest's value must not leak in
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "quickstart.py")],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-1500:]
    losses = [float(x) for x in re.findall(r"mean loss ([0-9.]+)", out.stdout)]
    assert len(losses) >= 3, out.stdout
    assert losses[-1] < 0.9 * losses[0], f"no descent: {losses}"


def test_the_readme_shows_the_example_as_it_is():
    """Every code line of the example appears in the README, in order. Prose can change;
    the code a user copies cannot differ from the code that is tested."""
    readme = (ROOT / "README.md").read_text()
    code = [l for l in (ROOT / "examples" / "quickstart.py").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith(("#", '"""'))]
    body = code[code.index("import os"):]
    at = 0
    for line in body:
        found = readme.find(line, at)
        assert found >= 0, f"README is missing or reorders this line of the example: {line!r}"
        at = found + len(line)
