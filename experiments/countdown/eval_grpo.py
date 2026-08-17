#!/usr/bin/env python
"""Held-out eval for GRPO checkpoints. Torch venv, same task.py, same tiers.

    <venv>/bin/python eval_grpo.py --config grpo.yaml

Walks checkpoint-*/ under the config's results_dir, plus the base model as step 0,
greedy-decodes the same held-out set the ES arms use (same eval_seed against the
same training pool, so the puzzles are byte-identical), and appends one line per
checkpoint to eval.jsonl in the results dir. Greedy, because the ES eval is greedy
and the comparison only means something if the decode rule is shared.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None,
                    help="results dir holding checkpoint-*/; defaults to the config's")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    out = args.results if args.results else HERE / cfg["results_dir"]

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
    train = task.make_puzzles(cfg["puzzle_seed"], cfg["n_puzzles"])
    ev = task.make_eval_puzzles(cfg["eval_seed"], cfg["eval_puzzles"], train)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p.prompt()}],
                                tokenize=False, add_generation_prompt=True)
        for p in ev
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EB = cfg["eval_batch"]
    eval_file = out / "eval.jsonl"

    ckpts = [(0, cfg["model"])] + sorted(
        ((int(p.name.split("-")[1]), p) for p in out.glob("checkpoint-*")),
    )
    for step, src in ckpts:
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16).to(device).eval()
        rewards = np.zeros(len(ev), np.float32)
        for lo in range(0, len(prompts), EB):
            batch = tok(prompts[lo:lo + EB], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
            with torch.no_grad():
                # repetition_penalty=1.0 explicitly: the checkpoint ships a
                # generation_config.json with repetition_penalty 1.1, and HF generate
                # applies it even under do_sample=False. Measured on 12 prompts, the
                # penalty flips the first content token on every one (it penalizes
                # '>' because '>' appears in the prompt), which is a different decode
                # rule than the ES arm's plain argmax and made the two arms'
                # baselines on identical weights differ (0.105 vs 0.054).
                gen = model.generate(**batch, max_new_tokens=cfg["max_new"],
                                     do_sample=False, repetition_penalty=1.0,
                                     pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for j, (t, p) in enumerate(zip(texts, ev[lo:lo + EB])):
                rewards[lo + j] = task.reward(t, p)
        rec = {"step": step, "eval_reward": float(rewards.mean()),
               "eval_solved": float((rewards == 1.0).mean()),
               "eval_format": float((rewards >= 0.1).mean())}
        with eval_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
