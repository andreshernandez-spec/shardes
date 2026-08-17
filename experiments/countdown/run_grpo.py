#!/usr/bin/env python
"""E13, the GRPO reference arm. TRL, in its own venv, sharing task.py with the ES arm.

    python -m venv .venv-grpo && .venv-grpo/bin/pip install -r requirements-grpo.txt
    .venv-grpo/bin/python run_grpo.py --config grpo.yaml
    .venv-grpo/bin/python run_grpo.py --config grpo.yaml --dry-run   # no training step

Same checkpoint, same puzzle generator and seed, same reward tiers through the same
parser: the two arms differ in the optimizer and nothing else they can help. This file
imports task.py and never shardes; the ES arm imports shardes and never torch. The venv
split exists so torch's CUDA stack and JAX's never share a process or an install.

**The hyperparameters are placeholders until transcribed from Qiu et al.** grpo.yaml
marks each value that must come from their paper's GRPO baseline table. E13's baseline
discipline (docs/05) is their published settings, cited, untuned by us in either
direction; a run from untranscribed placeholders is a smoke test, not the reference arm,
and the log records the config verbatim so nobody can mistake one for the other.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task  # noqa: E402  - the shared reward; the point of the file


def build_dataset(cfg, tokenizer):
    from datasets import Dataset  # noqa: PLC0415

    puzzles = task.make_puzzles(cfg["puzzle_seed"], cfg["n_puzzles"])
    rows = [
        {"prompt": [{"role": "user", "content": p.prompt()}],
         "numbers": list(p.numbers), "target": p.target}
        for p in puzzles
    ]
    return Dataset.from_list(rows)


def reward_fn(completions, numbers, target, **kwargs):
    """TRL calls this with the generation group; scores through the shared parser."""
    out = []
    for completion, nums, tgt in zip(completions, numbers, target):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        out.append(task.reward(text, task.Puzzle(tuple(nums), tgt)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, score one canned completion, train nothing")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    ds = build_dataset(cfg, tok)

    if args.dry_run:
        row = ds[0]
        canned = f"<answer>{row['numbers'][0]} + {row['numbers'][1]}</answer>"
        score = reward_fn([canned], [row["numbers"]], [row["target"]])
        print(f"dataset rows: {len(ds)}; canned completion scored {score[0]} "
              f"(0.1 = well-formed wrong, the expected value)")
        print(json.dumps({k: cfg[k] for k in sorted(cfg)}, indent=1))
        return 0

    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    out = HERE / cfg["results_dir"]
    targs = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=cfg["prompts_per_step"],
        num_generations=cfg["group_size"],
        max_prompt_length=cfg["pad_to"],
        max_completion_length=cfg["max_new"],
        learning_rate=cfg["lr"],
        beta=cfg["kl_beta"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        max_steps=cfg["steps"],
        seed=cfg["seed"],
        logging_steps=1,
        save_steps=cfg["checkpoint_every"],
        report_to=[],
        bf16=cfg.get("bf16", True),
    )
    trainer = GRPOTrainer(
        model=cfg["model"], args=targs, train_dataset=ds, reward_funcs=reward_fn,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
