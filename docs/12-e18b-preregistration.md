# 12 — E18/E18b: what the cost model predicts, written before the rental

Written 2026-08-23, before any 2-node cluster exists. `docs/10` §2 froze four directional
hypotheses (H1-H4) and `e18b.yaml` froze four more (G1-G4), and both H4 and G4 say the
same thing: a calibrated model should predict the sign before the cells run. Until now
there was no model to run, only a bracket, because the contraction had never been timed.
`docs/11-cost-model.md` is the model; this is its output, frozen.

Regenerate with `python experiments/phase2/multihost/preregister_e18b.py`. Nothing in it
reads anything that did not exist at this commit.

## The prediction

    delta(D)   = t_B - t_A
    delta(16)  = delta(8) + [alpha_inter + 4P/beta - ar_nvlink(4P)] - C/16
    flip at    beta* = 4P / (-delta(8) + ar_nvlink(4P) + C/16 - alpha_inter)

`delta(8)` is the committed single-node A100 sweep, `ar_nvlink` the measured NVLink
all-reduce at the same payload (`results-ladder`), `C` the replicated contraction solved
from those two, `alpha_inter` a 100 us placeholder the preflight will replace.

predicted `delta_16` in ms; negative = B wins, positive = A wins

| cell | delta_8 | C used | flip beta | 1gbit | 10gbit | socket 2 GB/s | 25 GbE | 100 GbE | 200Gb IB |
|---|---|---|---|---|---|---|---|---|---|
| seed_regenerated d=512 N=1024 | -25.92 | 29.72 | 0.23 GB/s | +22.54 | -22.76 | -24.65 | -25.78 | -27.29 | -27.54 |
| seed_regenerated d=2048 N=128 | -21.99 | 26.16 | 4.12 GB/s | +780.85 | +56.08 | +25.88 | +7.76 | -16.40 | -20.43 |
| seed_regenerated d=2048 N=256 | -54.30 | 63.08 | 1.70 GB/s | +746.24 | +21.46 | -8.74 | -26.86 | -51.02 | -55.04 |
| mirrored_lr1 d=512 N=1024 | +0.08 | 0.01 | no flip | +50.39 | +5.09 | +3.20 | +2.07 | +0.56 | +0.31 |
| mirrored_lr1 d=2048 N=128 | +1.63 | 0.00* | no flip | +806.11 | +81.34 | +51.14 | +33.02 | +8.86 | +4.83 |
| mirrored_lr1 d=2048 N=256 | +1.40 | 0.00* | no flip | +805.89 | +81.11 | +50.91 | +32.79 | +8.63 | +4.61 |

`*` C solved negative (the open term in `docs/11` §4) and clamped to zero.

## What this commits us to

**Three ordered flip bandwidths, 0.23, 1.70 and 4.12 GB/s.** They are ordered by how much
contraction work the cell gives B to save per byte it must move, and the ordering is the
part of the prediction that survives every caveat below. A sweep that flips them, or that
finds one cell flipping outside 0.1-10 GB/s, falsifies the model and not just a constant.

**H3 and G3 are predicted with a large margin.** `mirrored_lr1` is A-favored at every
fabric and every cell, from +0.3 ms on 200Gb IB to +806 ms on 1 gbit. Nothing marginal
here; if a `mirrored_lr1` cell comes back B-favored across a host boundary, the model is
wrong in a way no caveat covers.

**G2 is predicted to be wrong as written for one cell.** `e18b.yaml` G2 says
`seed_regenerated` at d=2048 "stays B-favored at socket-native but the sign FLIPS to A at
some throttled beta". The model says d=2048 N=128 flips at 4.12 GB/s, so if socket-native
measures below that it is ALREADY A-favored unthrottled and there is no B-favored anchor
to throttle away from. The d=2048 N=256 cell flips at 1.70 GB/s and behaves as G2
describes for a socket-native above that. Both readings are written down here; the
preflight's measured socket-native beta decides which, and that decision is a result.

**H2 is predicted to hold only on a fast fabric.** `seed_regenerated` keeps B at D=16 on
100 GbE and IB and loses it below about 4 GB/s at d=2048. H2 was stated without a
bandwidth qualifier; this adds one before the measurement rather than after.

## What could make it wrong

1. **C is solved, not measured.** `contraction_isolation.py` exists and has not run on an
   A100. It runs in the preflight, `predict.py` picks up the measured value, and any
   disagreement with the C column above is reported against this file. The signs are not
   sensitive: on the seed cells C/16 is 1.6-3.9 ms against boundary penalties of tens to
   hundreds of ms, and on the low-rank cells C is clamped to zero, which favors A, and A
   is predicted to win by 5-800 ms anyway.
2. **Nominal rates are line rates.** A ring all-reduce puts roughly `2(D-1)/D` times the
   payload on the wire. A beta fitted from a measured all-reduce absorbs that; a nominal
   "10gbit" does not. The throttled predictions can be optimistic by up to about 2x, which
   scales all three flip bandwidths together and leaves their ordering intact.
3. **alpha_inter is a guess.** At 4P = 96 MiB the bandwidth term is 4 ms even at 25 GB/s,
   so 50 us or 500 us changes no sign in the table.
4. **The single-node anchor assumes eval cancels.** `how` changes what is communicated and
   not what is computed, so `ask` and `apply` are identical between placements. The 1x8
   topology re-measures the anchor on the rented node and will show if that stopped being
   true.

## Order of operations on the cluster

The claim that these are predictions rests entirely on this order, and the session log's
timestamps are the witness:

1. preflight: fabric probe, psum ladders per topology, and `contraction_isolation.py` on
   the 1x8 mesh;
2. `predict.py` and `predict_e18b.py`, writing `predictions.json` and
   `predictions-e18b.json` with the measured C;
3. only then the timed 2x4, 2x8 and throttled cells.

A cell timed before step 2 is a measurement, not a test of a prediction, and gets reported
as one.
