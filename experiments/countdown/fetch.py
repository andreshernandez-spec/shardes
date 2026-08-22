#!/usr/bin/env python
"""Fetch the E13 checkpoint into the HF cache. The one place E13 touches the network.

    python fetch.py

Everything else (the golden tests, the task, the pilot bootstrap) reads the cache with
`local_files_only=True` and skips or fails loudly when this has not been run. Weights
and tokenizer only: no torch, no transformers needed to fetch.

Qwen2.5-0.5B-Instruct rather than the base model, and the choice is recorded here
because it is a fairness decision, not a convenience: at N=30 with a verifiable reward,
the pilot needs completions that parse at generation zero, and the instruct model's
answers do while the base model's often do not. Qiu et al. fine-tune instruct-class
models on Countdown as well. Both arms (ES and GRPO) get the same checkpoint, so the
comparison is unaffected; absolute reward numbers are not comparable to papers that
start from base models, and docs/05 E13 says so.
"""

from huggingface_hub import snapshot_download

import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"

if __name__ == "__main__":
    path = snapshot_download(
        REPO,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
    )
    print(f"{REPO} -> {path}")
