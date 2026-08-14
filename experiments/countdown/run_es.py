#!/usr/bin/env python
"""E13, the ES arm: Countdown on Qwen2.5-0.5B-Instruct.

    python run_es.py --config pilot.yaml
    python run_es.py --config pilot.yaml --smoke   # tiny overrides, laptop-sized

One generation: ask -> every member greedily decodes every puzzle -> the verifiable
reward, on the host, through the parser in task.py -> tell on -mean(reward), because
`tell` descends and reward ascends.

The reward is computed on the host, outside jit, and that is a design fact rather than
a shortcut: the fitness is *text* passed through an exact parser, and the whole point
of a verifiable reward is that it is not a differentiable, jittable surrogate. The
device work per generation is `es.apply`'s decode; the host work is regex and Fraction
arithmetic on n * puzzles short strings.

Fitness is mean reward over the generation's puzzle batch, f32 on the host (`tell`
refuses narrower). Puzzles rotate deterministically through a seed-fixed list, every
member sees the same puzzles (common random numbers, same reasoning as `apply`
replicating x), and both E13 arms consume the same puzzle sequence.

Resume: the master params and the generation counter are checkpointed every
`checkpoint_every` generations to `<out>/state.npz`; the log is JSONL, one line per
generation, appended. Restarting replays nothing and re-derives the same future
population keys from the counter, which is what `State.generation` exists for.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import task  # noqa: E402

from shardes import sharding  # noqa: E402
from shardes.core import ShardedES  # noqa: E402
from shardes.problems import qwen2  # noqa: E402
from shardes.strategies.lowrank import LowRank  # noqa: E402
from shardes.strategies.mirrored import Mirrored  # noqa: E402
from shardes.strategies.seed_regenerated import SeedRegenerated  # noqa: E402

OUTPUTS = ("results-es",)

STRATEGIES = {
    "mirrored_seed": lambda cfg: Mirrored(SeedRegenerated()),
    "mirrored_lr1": lambda cfg: Mirrored(LowRank(r=1)),
    "mirrored_lr4": lambda cfg: Mirrored(LowRank(r=4)),
    "mirrored_lr16": lambda cfg: Mirrored(LowRank(r=16)),
}


def tokenize(tok, puzzles, pad_to):
    """Chat-templated prompts, right-padded to one length. Returns ids, plen."""
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p.prompt()}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in puzzles
    ]
    encs = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    plen = np.array([len(e) for e in encs], np.int32)
    if plen.max() > pad_to:
        raise ValueError(f"prompt of {plen.max()} tokens exceeds pad_to={pad_to}")
    ids = np.zeros((len(encs), pad_to), np.int32)
    for i, e in enumerate(encs):
        ids[i, : len(e)] = e
    return jnp.asarray(ids), jnp.asarray(plen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="4 members, 2 puzzles, 24 new tokens, 2 generations: end-to-end "
                         "wiring on a laptop, never a result")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(population=4, puzzles_per_gen=2, max_new=24, generations=2,
                   results_dir=cfg["results_dir"] + "-smoke")

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    ckpt = snapshot_download(cfg["model"], local_files_only=True,
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(ckpt)
    mcfg = qwen2.Config.qwen25_05b()
    params = qwen2.load(ckpt, mcfg, dtype=jnp.bfloat16)

    mesh = sharding.make_mesh(cfg.get("devices"))
    es = ShardedES(
        STRATEGIES[cfg["strategy"]](cfg), n=cfg["population"], sigma=cfg["sigma"],
        lr=cfg["lr"], mesh=mesh, compute_dtype=jnp.bfloat16,
    )

    out = HERE / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    state_file, log_file = out / "state.npz", out / "log.jsonl"

    state = es.init(jax.random.key(cfg["seed"]), params)
    if state_file.exists():
        saved = np.load(state_file, allow_pickle=False)
        flat, treedef = jax.tree.flatten(state.params)
        state = state._replace(
            params=jax.tree.unflatten(treedef, [jnp.asarray(saved[f"p{i}"]) for i in range(len(flat))]),
            generation=jnp.int32(saved["generation"]),
        )
        print(f"resumed at generation {int(state.generation)}")

    puzzles = task.make_puzzles(cfg["puzzle_seed"], cfg["n_puzzles"])
    env = harness.capture_env(HERE, (*OUTPUTS, cfg["results_dir"]))
    B, T = cfg["puzzles_per_gen"], cfg["pad_to"] + cfg["max_new"]

    def model(p, batch):
        ids, plen = batch
        return qwen2.generate(p, ids, plen, mcfg, cfg["max_new"])

    eos = tok.eos_token_id
    while int(state.generation) < cfg["generations"]:
        g = int(state.generation)
        batch_puzzles = [puzzles[(g * B + j) % len(puzzles)] for j in range(B)]
        ids, plen = tokenize(tok, batch_puzzles, cfg["pad_to"])

        t0 = time.perf_counter()
        pert, state = es.ask(state)
        gen = np.asarray(es.apply(model, state, pert)((ids, plen)))  # (n, B, T)
        rewards = np.zeros((cfg["population"], B), np.float32)
        for m in range(cfg["population"]):
            for j in range(B):
                row = gen[m, j, int(plen[j]):]
                stop = np.where(row == eos)[0]
                text = tok.decode(row[: stop[0]] if len(stop) else row)
                rewards[m, j] = task.reward(text, batch_puzzles[j])
        fitness = jnp.asarray(-rewards.mean(axis=1), jnp.float32)  # tell descends
        state = es.tell(state, pert, fitness)
        dt = time.perf_counter() - t0

        rec = {"generation": g, "seconds": dt,
               "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
               "solved_frac": float((rewards == 1.0).mean()),
               "format_frac": float((rewards >= 0.1).mean()), "env": None}
        if g == 0:
            rec["env"] = env  # once per file; every line repeating it doubles the log
        with log_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{g}] reward {rec['reward_mean']:.3f} max {rec['reward_max']:.1f} "
              f"format {rec['format_frac']:.2f} solved {rec['solved_frac']:.2f} {dt:.1f}s",
              flush=True)

        if (g + 1) % cfg.get("checkpoint_every", 10) == 0 or g + 1 == cfg["generations"]:
            flat, _ = jax.tree.flatten(jax.device_get(state.params))
            np.savez(state_file, generation=int(state.generation),
                     **{f"p{i}": leaf for i, leaf in enumerate(flat)})
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
