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

A cell UNDER 100% is a cell where B is slower than `C/D + ar` allows, and a contraction
cannot cost less than nothing. Eight of the eleven A-favored cells are under it, four per
platform, and they solve to a negative `C`. Every one is low-rank; six of the eight are at
d=2048. The shortfall runs from 0.01 ms up to 0.71 ms on the A100 and 0.63 ms on the v5e,
both worst at `mirrored_lr1` d=2048 N=128. `results-contraction/README.md` lists the three candidate causes and how the
measurement separates them. Until it runs, the low-rank side of the crossover is bracketed
rather than derived, and §8 says so in those words.

## 5. Where it became a prediction, and how it did

**Tested 2026-08-24 on the E18 cluster, and it missed.** `contraction_isolation.py` ran in
the preflight, so `predict.py` made point predictions with a measured `C` rather than the
old bracket, before any boundary cell. Outcome:

- Direction right. Every low-rank cell predicted A and measured A; the boundary hurts B.
- `C` itself is sound. Measured 28.7, 31.9 and 64.0 ms on the seed cells against 29.7,
  26.2 and 63.1 ms solved backwards from the D=8 grid.
- **The fabric term is what broke.** `comm_bump` predicted 10.1 ms for the model-sized
  all-reduce across the boundary; the measurement is about 108 ms. Ten times, where
  caveat 2 of `docs/12` allowed about two for the ring wire factor. So the flip cell
  (seed_regenerated d=2048) was predicted B and measures A.

The lesson is specific and worth keeping: `alpha + 4P/beta` with a beta from a preflight
ladder does not price a real collective on a socket fabric. Between the ladder's payload
and a generation's psum sit the transport, the message chunking and the contention, and on
NVLink those cost a factor of one while on TCP sockets they cost a factor of ten. A cost
model calibrated on a fat fabric extrapolates its own regime, not the next one.

## 6. The original plan (for the record)

Calibrated on the same platforms it explains, this is an account, not a forecast. It
becomes a forecast on a fabric it has not seen, which is E18/E18b: `docs/10` H4 and G4
already require predictions written before the throttled cells run, and
`multihost/predict.py` and `multihost/predict_e18b.py` currently bracket the contraction
between 0 and the whole measured gap because no measured `C` exists. With `C` measured they
become point predictions. That is the one experiment that closes two limitations at once,
the fat fabric and the cost model, and it costs one file written before a rental that is
already planned.

## 7. Files

| file | what |
|---|---|
| `experiments/phase2/allreduce_ladder.py` | `ar`, `ag`, alpha and beta per fabric (run: A100, v5e) |
| `experiments/phase2/contraction_isolation.py` | `C`, `C/D`, the in-situ all-reduce (pending) |
| `experiments/phase2/timemodel.py` | the model, forwards and backwards, per cell |
| `experiments/phase2/tb7.py` | the paper table |
| `experiments/phase2/regen_decompose.py` | where seed regeneration's time goes (A100 run, v5e pending) |
