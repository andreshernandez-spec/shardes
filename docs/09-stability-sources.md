# Stability of ES vs RL updates: sources (research spike, 2026-08-20)

Deep-research spike commissioned for E14/C7 and the paper's motivation paragraph.
Three parallel literature sweeps (RL instability theory; ES stability theory;
experimental evidence both ways), primary sources verified by search or direct
fetch. Citation keys match `paper/references.bib`. Verification caveats at the end.

## 1. Why RL's stabilization machinery exists (theory)

The machinery Andres described (EWMA'd weights, slow-shift constraints, tuned
schedules) is not folklore; each piece is the field's answer to a named theorem
or counterexample.

- **The deadly triad.** Function approximation + bootstrapping + off-policy
  learning can diverge, not merely converge slowly: Baird's star counterexample
  [baird1995], the convergence/divergence dichotomy of Tsitsiklis and Van Roy
  [tsitsiklis1997] (on-policy TD converges because the projected Bellman operator
  contracts under the on-policy distribution; off-policy weighting breaks it),
  Sutton and Barto ch. 11 [sutton2018] naming the triad, van Hasselt et al.
  [vanhasselt2018] measuring "soft divergence" in deep agents.
- **Target networks and Polyak/EWMA averaging are the fix for the triad.** DQN's
  frozen target breaks the feedback loop between the estimator and its own
  regression target [mnih2015]; DDPG makes it an explicit EWMA with tau << 1
  [lillicrap2015]; Zhang et al. [zhang2021target] PROVE a (regularized) target
  network restores convergence where semi-gradient TD provably diverges. "Only
  allow very slow and gradual shifts of the weights" is, verbatim, the theory's
  prescription.
- **Trust regions: why unconstrained policy steps are destructive.** The
  surrogate objective is estimated under the OLD policy's state distribution;
  its validity decays with policy divergence. Conservative policy iteration
  [kakade2002] and TRPO's monotonic-improvement bound (improvement >= surrogate
  minus C times max-KL) [schulman2015trpo] make small steps provably safe and
  large steps provably unsafe; PPO's clip is the cheap approximation
  [schulman2017ppo], and it demonstrably does not enforce the bound it
  approximates [engstrom2020].
- **The KL anchor in LLM RL is what makes the problem well-posed.** Naive reward
  maximization over a distribution collapses onto degenerate high-reward outputs;
  KL-regularized RL is exactly variational inference toward the tilted reference
  posterior [korbak2022]. Reward hacking is generic, not incidental: nontrivial
  unhackable proxies essentially do not exist [skalse2022], and gold reward rises
  then falls as a function of KL distance from the initial policy
  [gao2023overoptimization]. This is the theory behind E14's kl_beta arm.
- **Entropy collapse has a mechanism.** Per-step entropy change is governed by
  the covariance of log-probability and advantage: confident right answers get
  reinforced, mechanically draining entropy, with an empirical exchange law
  R = -a e^H + b [cui2025entropy]; the PPO/GRPO clip asymmetry itself biases
  entropy downward [clipentropy2025]. This names what E13's GRPO seed 1 did.
- **Score-function variance is the original sin** [williams1992]; baselines and
  advantages are control variates with a closed-form optimum [greensmith2004];
  GRPO's group-mean baseline is a direct application.
- **Curricula: the weakest theory.** Frameworks and surveys [narvekar2020,
  portelas2020]; the hardness root is exploration lower bounds (sparse-reward
  exploration needs guidance that is information-theoretically unavailable to
  undirected search). Necessity theorems are scarce; the evidence is mostly the
  systems record (section 3).

## 2. Why ES updates should be stable (theory), and the price

- **ES optimizes a Gaussian-smoothed objective** [nesterov2017]: always
  differentiable, Lipschitz gradient controlled by sigma; jagged or chaotic
  landscapes become optimizable. The price is a theorem too: dimension-dependent
  iteration penalties with matching information-theoretic lower bounds
  [duchi2015]. Any LLM-scale ES claim lives inside that tension; Qiu et al. only
  speculate (low intrinsic dimensionality) about why N=30 works in billions of
  dimensions [qiu2025], and the sharpest available account is that ES progresses
  on a low-dimensional task subspace while drifting broadly off-task
  [hoy2026geometry], which explains the success and prices it.
- **Rank shaping is invariance by design.** NES [wierstra2014] introduces rank
  shaping explicitly for invariance to monotone reward transformations; CMA-ES
  doctrine [hansen2016] and the IGO framework [ollivier2017] DERIVE the update
  family from invariance principles. Reward scale, shifts and outliers cannot
  blow up the step. This is the mechanism C7 claims, stated as such.
- **No bootstrapping, no per-step credit assignment** [salimans2017]: invariance
  to action frequency and delayed reward, no discount, no value function; the
  whole deadly triad is absent by construction. Paid for in samples (3-10x A3C
  on Atari, their own numbers).
- **Chaos.** Exact gradients through long unrolls explode when the Jacobian
  spectrum exceeds one, becoming formally correct and useless [metz2021]; the
  likelihood-ratio family stays robust [parmas2018]; smoothed objectives train
  where exact ones cannot [metz2019, suh2022].
- **Few hyperparameters** is a real tradition (CMA-ES quasi-parameter-free
  [hansen2001]; ARS's handful [mania2018]; Qiu's single fixed set vs
  per-experiment RL sweeps [qiu2025]) but see section 4.

## 3. The experimental record

- **RL brittleness, measured**: same algorithm, two groups of 5 seeds,
  statistically distinguishable (t=-9.09, p=0.0016); an architecture change flips
  PPO's HalfCheetah score from +2201 to -1180 [henderson2018]. Code-level tricks
  contribute more reward than the choice of algorithm [engstrom2020]. 250k-agent
  study: obscure choices (final-layer init scale) have outsized effects; value
  normalization helps or hurts depending on environment [andrychowicz2020].
- **LLM RL in practice**: InstructGPT needed the KL anchor plus a pretraining-mix
  term to stop regressions [ouyang2022]; Llama 3 dropped PPO for DPO citing
  stability and scale [llama3]; DeepSeek-R1 abandoned neural reward models
  because they "inevitably" get hacked and needed a hand-built multi-stage
  pipeline [deepseekr1]; naive GRPO reached 30 vs 47 AIME points until four
  targeted fixes [dapo2025]; GRPO has a structural length bias [drgrpo2025];
  under standardized evaluation most RL reasoning gains shrink and overfit
  [sober2025].
- **The curricula record**: OpenAI Five's twenty "surgeries", annealed game
  mechanics with explicit revert-and-retry, data-reuse cliff at 8x [openai5];
  AlphaStar's league because self-play cycles [alphastar]; ADR because manual
  randomization tuning stopped scaling, 1.8 vs 26.8 real-robot successes
  [akkaya2019].
- **ES robustness**: one fixed hyperparameter set across 51 Atari games
  [salimans2017]; ARS with linear policies matching MuJoCo SOTA over 100 seeds
  [mania2018]; random search beating A3C on 6 games [such2017]; and at LLM scale,
  ES reward std 15.5x lower than GRPO over 4 runs, no reward hacking even with
  zero KL penalty, one fixed ES setting against per-experiment RL sweeps
  [qiu2025]. Note what is NOT in the record: nobody has measured perturbation
  robustness (multiplied hyperparameters) at LLM scale on both sides. That is
  E14's gap to fill.

## 4. Counter-evidence, which the paper must cite

- **ES is conditionally robust, not unconditionally.** Salimans's own finding 1:
  without virtual batch normalization and reparameterization, "ES proved
  brittle" [salimans2017]. ARS succeeds on Walker2d in only 20% of runs and is
  as seed- and hyperparameter-sensitive as its targets on some tasks [mania2018].
- **ES reward-hacks too**: canonical ES found the Q*bert score bug
  [chrabaszcz2018]; Qiu's no-hacking result is task- and scale-specific.
- **ES gets trapped by deceptive and sparse rewards** [conti2017,
  chrabaszcz2018]: no directed exploration.
- **ES updates can be destructive in a different sense**: dense, large-norm
  updates cause catastrophic forgetting where GRPO's sparse ones do not
  [abdi2026forgetting]. Reward-variance stability and capability preservation
  are different claims; C7 claims only the first.
- **RL can be stable with the right recipe**: vanilla PPO with a learned critic
  and NO KL penalty trains stably at 32B and beats R1-Zero with a tenth of the
  steps [orz2025]. This is the single most important scoping fact for C7: the
  fragile object in our data and in the 2025 literature is critic-free
  GRPO-style RLVR, not RL-with-critic. C7 is scoped accordingly.
- **Sigma is the hyperparameter that remains**, and it changes what is being
  optimized [berahas2022]; self-adaptive step-size control can itself cause
  premature convergence [rudolph2001]. E14's sigma arm measures exactly this.

## 5. What the spike changes

1. **C7 rescoped** to "GRPO-style critic-free RLVR at this scale", with
   [orz2025] cited as the reason for the scoping (docs/08 updated).
2. **E14 gains a cheap measurement**: log policy entropy per step in the GRPO
   arms (TRL exposes it), so our collapse observations connect to the
   mechanism in [cui2025entropy] rather than floating free.
3. **The introduction paragraph** writes itself from section 1: the machinery
   is the field's institutionalized answer to theorems, ES lacks the machinery
   because it lacks the theorems' preconditions, and E14 tests whether that
   absence is load-bearing.
4. **Honest-limits paragraph** for section 7 comes from section 4, especially
   [abdi2026forgetting] and [orz2025].

## Verification caveats

Agents verified bibliographic existence for everything cited; depth varies.
Read-from-PDF: Salimans, Henderson, Engstrom, Berner, Akkaya, Qiu numbers.
Abstract-level: Nesterov-Spokoiny, Duchi, Berahas, Hansen, Rudolph, EGGROLL's
GRPO comparison. Listing-level only (2026 arXiv IDs; re-verify before citing in
the submitted PDF): [hoy2026geometry] arXiv:2604.01499, [abdi2026forgetting]
arXiv:2601.20861, [clipentropy2025] arXiv:2509.26114. AlphaStar's quantitative
league ablations are paywalled; only qualitative claims are cited.
