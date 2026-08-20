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
    "mirrored_seed": lambda cfg: Mirrored(SeedRegenerated(chunk=cfg.get("eval_chunk", 1))),
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
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed; the results dir gets a -sN suffix "
                         "so seeds never share a directory")
    ap.add_argument("--smoke", action="store_true",
                    help="4 members, 2 puzzles, 24 new tokens, 2 generations: end-to-end "
                         "wiring on a laptop, never a result")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg.update(seed=args.seed, results_dir=cfg["results_dir"] + f"-s{args.seed}")
    if args.smoke:
        cfg.update(population=4, puzzles_per_gen=2, max_new=24, generations=2,
                   eval_puzzles=4, eval_every=1, eval_batch=2, eval_chunk=1,
                   results_dir=cfg["results_dir"] + "-smoke")

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    ckpt = snapshot_download(cfg["model"], local_files_only=True,
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(ckpt)
    mcfg = qwen2.Config.qwen25_05b()
    params = qwen2.load(ckpt, mcfg, dtype=jnp.bfloat16)

    # C6c ablation: with freeze_embedding the embedding leaf never enters the ES
    # state. It is closed over the model function as a constant, so it is neither
    # perturbed nor updated and the strategy's d_eff drops accordingly; the tied
    # table still serves both seams (embed in, head out) through `frozen`. The
    # comparison arm is the ordinary run, where the same leaf lives in the state.
    frozen = {}
    if cfg.get("freeze_embedding"):
        frozen["embedding"] = params.pop("embedding")

    def with_frozen(p):
        return {**p, **frozen} if frozen else p

    mesh = sharding.make_mesh(cfg.get("devices"))
    es = ShardedES(
        STRATEGIES[cfg["strategy"]](cfg), n=cfg["population"], sigma=cfg["sigma"],
        lr=cfg["lr"], mesh=mesh, compute_dtype=jnp.bfloat16,
    )

    out = HERE / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    state_file, log_file = out / "state.npz", out / "log.jsonl"
    eval_file = out / "eval.jsonl"

    # E14: drift from the pretrained weights, logged with every eval. The reference
    # is theta_0 (the loaded checkpoint), never the resume point, so a resumed run
    # reports the same quantity an unbroken one would. Shares buffers with params
    # (JAX never mutates them), so the copy costs HBM only once updates diverge.
    init_ref = params

    @jax.jit
    def l2_from_init(p):
        sq = jax.tree.map(
            lambda a, b: jnp.sum((a.astype(jnp.float32) - b.astype(jnp.float32)) ** 2),
            p, init_ref)
        return jnp.sqrt(jax.tree.reduce(lambda x, y: x + y, sq))

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
        return qwen2.generate(with_frozen(p), ids, plen, mcfg, cfg["max_new"],
                              prefill=prefill_len)

    # Jitted once, traced once: without this the vmap/scan dispatches op by op and the
    # GPU idles between kernels. Measured: the un-jitted loop held ~330 s/generation on
    # the laptop with prefill making no difference, because dispatch, not compute, was
    # the bottleneck; the A100 probe's 600 s/generation was the same defect at scale.
    @jax.jit
    def evaluate(state, pert, ids, plen):
        return es.apply(model, state, pert)((ids, plen))

    # tell needs the same treatment, and it was the bigger half: un-jitted, its contract
    # scan re-lowered every generation (the per-generation key and params view enter as
    # constants, and the executable cache misses on fresh constants). The A100 phase
    # probe measured steady-state eval at 4.1 s and tell at ~314 s, gens 2 and 3 alike.
    # Jitted, the arrays are traced arguments and the compile happens once.
    tell = jax.jit(es.tell)

    eos = tok.eos_token_id
    # One prefill length for the whole run, the pool-wide minimum: a per-generation
    # value changes the prefill's static shape and forces a recompile every generation,
    # which the re-timed smoke measured as eating the entire prefill saving.
    _, all_plen = tokenize(tok, puzzles, cfg["pad_to"])
    prefill_len = int(all_plen.min())

    # Held-out evaluation: greedy decode of the MASTER weights (their bf16 view, the
    # same dtype the training forward runs in) on a set provably disjoint from the
    # training pool. Runs before the update at every eval_every-th generation, so
    # generation 0 is the pre-training baseline, and once more after the last update.
    # Same decode, same parser, same tiers as training; only the puzzles differ.
    eval_puzzles = task.make_eval_puzzles(cfg["eval_seed"], cfg["eval_puzzles"], puzzles)
    ev_ids, ev_plen = tokenize(tok, eval_puzzles, cfg["pad_to"])
    ev_prefill = int(ev_plen.min())
    EB = cfg["eval_batch"]
    if len(eval_puzzles) % EB:
        raise ValueError(f"eval_batch={EB} must divide eval_puzzles={len(eval_puzzles)}")

    @jax.jit
    def eval_decode(params, ids, plen):
        view = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
        return qwen2.generate(with_frozen(view), ids, plen, mcfg, cfg["max_new"],
                              prefill=ev_prefill)

    def run_eval(state):
        g = int(state.generation)
        t0 = time.perf_counter()
        rewards = np.zeros(len(eval_puzzles), np.float32)
        for lo in range(0, len(eval_puzzles), EB):
            gen = np.asarray(eval_decode(state.params, ev_ids[lo:lo + EB],
                                         ev_plen[lo:lo + EB]))
            for j in range(EB):
                row = gen[j, int(ev_plen[lo + j]):]
                stop = np.where(row == eos)[0]
                text = tok.decode(row[: stop[0]] if len(stop) else row)
                rewards[lo + j] = task.reward(text, eval_puzzles[lo + j])
        rec = {"generation": g, "eval_reward": float(rewards.mean()),
               "eval_solved": float((rewards == 1.0).mean()),
               "eval_format": float((rewards >= 0.1).mean()),
               "eval_seconds": time.perf_counter() - t0,
               "param_l2_from_init": float(l2_from_init(state.params))}
        with eval_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{g}] EVAL reward {rec['eval_reward']:.3f} solved {rec['eval_solved']:.3f} "
              f"format {rec['eval_format']:.2f} {rec['eval_seconds']:.0f}s", flush=True)

    while int(state.generation) < cfg["generations"]:
        g = int(state.generation)
        if g % cfg["eval_every"] == 0:
            run_eval(state)
        batch_puzzles = [puzzles[(g * B + j) % len(puzzles)] for j in range(B)]
        ids, plen = tokenize(tok, batch_puzzles, cfg["pad_to"])

        t0 = time.perf_counter()
        pert, state = es.ask(state)
        gen = np.asarray(evaluate(state, pert, ids, plen))  # (n, B, T)
        rewards = np.zeros((cfg["population"], B), np.float32)
        for m in range(cfg["population"]):
            for j in range(B):
                row = gen[m, j, int(plen[j]):]
                stop = np.where(row == eos)[0]
                text = tok.decode(row[: stop[0]] if len(stop) else row)
                rewards[m, j] = task.reward(text, batch_puzzles[j])
        fitness = jnp.asarray(-rewards.mean(axis=1), jnp.float32)  # tell descends
        state = tell(state, pert, fitness)
        dt = time.perf_counter() - t0

        rec = {"generation": g, "seconds": dt,
               "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
               "solved_frac": float((rewards == 1.0).mean()),
               "format_frac": float((rewards >= 0.1).mean()), "env": None}
        if g == 0:
            rec["env"] = env  # once per file; every line repeating it doubles the log
            rec["config"] = dict(cfg)  # seed overrides and all, so the log is the record
        with log_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{g}] reward {rec['reward_mean']:.3f} max {rec['reward_max']:.1f} "
              f"format {rec['format_frac']:.2f} solved {rec['solved_frac']:.2f} {dt:.1f}s",
              flush=True)

        if (g + 1) % cfg.get("checkpoint_every", 10) == 0 or g + 1 == cfg["generations"]:
            flat, _ = jax.tree.flatten(jax.device_get(state.params))
            np.savez(state_file, generation=int(state.generation),
                     **{f"p{i}": leaf for i, leaf in enumerate(flat)})
    run_eval(state)  # the post-training point the whole curve exists for
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
