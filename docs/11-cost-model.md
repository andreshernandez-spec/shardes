# 11 — The contraction crossover as a time model

Written 2026-08-23, after review 7's second hole. The paper predicted the crossover in
bytes (§4, committed before E4 and E7) and measured it in seconds (§6), and nothing joined
the two. This is the join, what it explains, and what it does not.

## 1. Bytes are not the wrong attribution

They are the right thing to commit in advance. A byte count is a property of the
placement: strategy A moves `8N`, strategy B moves `4P + 4N`, and neither number depends on
the machine. That is why §4 could be written before the runs and corrected by them
(`tb2`), and it should stay exactly as it is.

Time is what decides a cell, and every headline number in §6 already is time. So the paper
never attributed costs in bytes; it predicted in bytes and measured in time. The hole was
the missing middle: nothing converted one into the other, so

> B all-reduces a model-sized buffer and still wins whenever the contraction it avoids
> costs more than those bytes

was an assertion. It cannot follow from bytes, because bytes reach time through an achieved
rate that is itself a function of message size, and the grids already contain the
counterexample: identical byte counts, different crossover on the A100 and the v5e. The
"absolute numbers are platform-specific" limitation and the "bytes, not time" limitation
were the same hole written twice.

## 2. The model

With `P = 6 d^2` for the block, `D` devices, `C(N, d)` the replicated contraction, and
`ar`/`ag` the two collectives at their real payloads:

    t_A = eval + C(N, d)     + 2 ag(4N)
    t_B = eval + C(N, d) / D +   ag(4N) + ar(4P)

    t_A - t_B = C(N, d) (D - 1) / D + ag(4N) - ar(4P)

`eval` cancels: `how` changes what is communicated and not what is computed
(`core.py`), so `ask` and `apply` are identical between the placements and the difference
isolates the contraction. The crossover is where the collective B adds equals the
contraction B avoids:

    ar(4P) = C(N, d) (D - 1) / D + ag(4N)

Nothing here is fitted. `ar` and `ag` come from `allreduce_ladder.py`, `C` from
`contraction_isolation.py`, and the sweep is the thing being predicted.

## 3. What it explains

`timemodel.py` runs it against the committed D=8 grids.

- The dense side is the contraction, not the transfer. Where B wins, its model-sized
  all-reduce is 0.7 to 3.6% of the advantage it buys.
- The low-rank side is the transfer and almost nothing else. Where A wins, the same
  all-reduce is 57 to 155% of B's deficit.
- The platform shift falls out of one number. The 96 MiB payload costs 0.93 ms on NVLink
  and 2.01 ms on ICI, 2.17x, and A's low-rank lead grows from 4-15% to 17-46% in the same
  cells. The bytes did not move; the fabric did.

## 4. What it does not explain

Three cells past 100% are three cells where B is slower than `C/D + ar` allows, by 0.6 to
0.7 ms, and a contraction cannot cost less than nothing. Run backwards, four of twenty
A100 cells and four of sixteen v5e cells solve to a negative `C`, all of them low-rank at
d=2048. `results-contraction/README.md` lists the three candidate causes and how the
measurement separates them. Until it runs, the low-rank side of the crossover is bracketed
rather than derived, and §8 says so in those words.

## 5. Where it becomes a prediction

Calibrated on the same platforms it explains, this is an account, not a forecast. It
becomes a forecast on a fabric it has not seen, which is E18/E18b: `docs/10` H4 and G4
already require predictions written before the throttled cells run, and
`multihost/predict.py` and `multihost/predict_e18b.py` currently bracket the contraction
between 0 and the whole measured gap because no measured `C` exists. With `C` measured they
become point predictions. That is the one experiment that closes two limitations at once,
the fat fabric and the cost model, and it costs one file written before a rental that is
already planned.

## 6. Files

| file | what |
|---|---|
| `experiments/phase2/allreduce_ladder.py` | `ar`, `ag`, alpha and beta per fabric (run: A100, v5e) |
| `experiments/phase2/contraction_isolation.py` | `C`, `C/D`, the in-situ all-reduce (pending) |
| `experiments/phase2/timemodel.py` | the model, forwards and backwards, per cell |
| `experiments/phase2/tb7.py` | the paper table |
| `experiments/phase2/regen_decompose.py` | where seed regeneration's time goes (A100 run, v5e pending) |
