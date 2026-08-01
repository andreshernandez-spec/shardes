"""Push a kernel to Kaggle, wait for it, print its log.

    python experiments/phase1/kaggle/run.py --accelerator NvidiaTeslaT4

Needs ~/.kaggle/kaggle.json (Kaggle: Settings -> API -> Create New Token), chmod 600.

The push is staged in a temp directory with the username filled in, and the committed
metadata keeps the USERNAME placeholder. The Kaggle account is a personal one and not the
identity this repo is published under (docs/06), so it does not belong in git history,
where it cannot be taken back out.

Why a script and not three CLI calls: `kernels push` returns immediately, and the
interesting part is the log, which lives behind `kernels output` and only after the run
reaches a terminal state. Polling by hand invites reading a stale log from the previous
run and believing it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
CREDS = pathlib.Path.home() / ".kaggle" / "kaggle.json"
#: Matched case-insensitively. The CLI prints `KernelWorkerStatus.COMPLETE`, not `complete`,
#: and a lowercase comparison polls a finished kernel until the timeout.
TERMINAL = {"complete", "error", "cancelacknowledged"}


def username() -> str:
    """Ask the CLI, not the credential file.

    There are two credential shapes in the wild: the old `~/.kaggle/kaggle.json` with a
    username field, and `~/.kaggle/access_token`, which has no username in it at all.
    `kaggle config view` reports the resolved account either way, so read that and stay out
    of the credential format business.
    """
    for line in kaggle("config", "view").splitlines():
        if "username:" in line:
            return line.split("username:", 1)[1].strip()
    sys.exit(
        "no Kaggle credentials. Kaggle -> Settings -> API -> Create New Token, then put it "
        f"in {CREDS.parent}/ (kaggle.json or access_token) with chmod 600."
    )


def kaggle(*args: str) -> str:
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=["probe", "t2prime"], help="subdirectory to push")
    ap.add_argument(
        "--accelerator",
        default="NvidiaTeslaT4",
        help="NvidiaTeslaT4 gives TWO T4s, measured 2026-08-01 by the probe kernel. The API "
        "documents no count, so this was settled by running it.",
    )
    ap.add_argument("--poll", type=int, default=20, help="seconds between status checks")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    src = HERE / args.kernel
    meta = json.loads((src / "kernel-metadata.json").read_text())
    slug = meta["id"].split("/")[-1]
    meta["id"] = f"{username()}/{slug}"

    stage = pathlib.Path(tempfile.mkdtemp(prefix="shardes-kaggle-"))
    shutil.copy(src / meta["code_file"], stage)
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"pushing {meta['id']} (accelerator={args.accelerator})")
    print(kaggle("kernels", "push", "-p", str(stage), "--accelerator", args.accelerator))

    deadline = time.monotonic() + args.timeout
    while True:
        status = kaggle("kernels", "status", meta["id"]).strip()
        print(f"  {status}")
        if any(t in status.lower() for t in TERMINAL):
            break
        if time.monotonic() > deadline:
            sys.exit(f"still running after {args.timeout}s; check the notebook in the browser")
        time.sleep(args.poll)

    out = HERE / "output" / args.kernel  # per kernel, or the newest log wins the glob
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", meta["id"], "-p", str(out), "-o")
    logs = sorted(out.glob("*.log"))
    if not logs:
        sys.exit(f"no log in {out}; contents: {[p.name for p in out.iterdir()]}")
    for line in json.loads(logs[-1].read_text()):
        print(line.get("data", ""), end="")


if __name__ == "__main__":
    main()
