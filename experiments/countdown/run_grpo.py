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
    # The prompt budget the ES arm enforces in tokenize(), enforced here directly:
    # TRL removed GRPOConfig.max_prompt_length (gone by trl 1.10), and a truncated
    # prompt would be a silent task change anyway. Refuse instead.
    longest = max(
        len(tokenizer.apply_chat_template(r["prompt"], tokenize=True,
                                          add_generation_prompt=True))
        for r in rows
    )
    if longest > cfg["pad_to"]:
        raise ValueError(f"prompt of {longest} tokens exceeds pad_to={cfg['pad_to']}")
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
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed; the results dir gets a -sN suffix "
                         "so seeds never share a directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, score one canned completion, train nothing")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg.update(seed=args.seed, results_dir=cfg["results_dir"] + f"-s{args.seed}")

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
    # TRL's batch counts completions, not prompts, and must divide by
    # num_generations: one micro-batch is one prompt's whole group, and the
    # accumulation steps are the prompts. One optimizer step therefore consumes
    # prompts_per_step * group_size completions (240 at the committed config),
    # which is the sample-evaluation accounting grpo.yaml's steps are matched on.
    targs = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=cfg["group_size"],
        gradient_accumulation_steps=cfg["prompts_per_step"],
        num_generations=cfg["group_size"],
        max_completion_length=cfg["max_new"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["lr_schedule"],
        beta=cfg["kl_beta"],
        # E14: the ratio clip, explicit so the ablation arm can widen it. TRL's
        # default is 0.2; epsilon_high None means symmetric.
        epsilon=cfg.get("clip_epsilon", 0.2),
        epsilon_high=cfg.get("clip_epsilon_high"),
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

    # E14: theta_0 snapshot for the drift metric, f32 on CPU (~2 GB for 0.5B).
    # Taken from the trainer's own model so it is exactly what training mutates.
    import torch  # noqa: PLC0415  - the venv has it; the module header explains the split
    init_sd = {k: v.detach().to("cpu", torch.float32).clone()
               for k, v in trainer.model.state_dict().items()}

    trainer.train()

    # The full log history, whatever keys this TRL version emits (entropy,
    # completions/mean_length, kl, ...): recorded now, extracted at analysis,
    # so the metric list is not hostage to TRL's naming.
    with (out / "train-log.jsonl").open("w") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row) + "\n")

    sq = 0.0
    final_sd = trainer.model.state_dict()
    for k, v0 in init_sd.items():
        d = final_sd[k].detach().to("cpu", torch.float32) - v0
        sq += float((d * d).sum())
    summary = {"param_l2_from_init": sq ** 0.5, "config": dict(cfg)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"param_l2_from_init {summary['param_l2_from_init']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
