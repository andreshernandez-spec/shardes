# Why the low-rank arms run out of memory and the seed arm does not

`e17_memory_probe.py`, laptop CPU, 8 simulated devices, code at 83e602e. Diagnosis
only: the absolute figures are XLA's temporary estimate for a CPU executable on a
cut-down model and nothing cites them. The scalings are the result.

## The thing that needed explaining

E17b records 57 OOM cells out of 118. Every one is a low-rank arm; `mirrored_seed`
fits all 16 of its cells. E17 read that column as the storage-for-compute trade
("rank 1 does not fit one 16 GB chip even at N=32 while the seed arm runs at every
cell"), and `e17b.yaml` froze the prediction that "the low-rank arms store factors,
so higher rank fits fewer cells; rank 16 is expected to OOM where rank 1 runs at the
largest N".

The grid falsifies the prediction on its own. The three ranks OOM at exactly the same
27 cells, and their reported HBM temporaries agree to 0.1%: 114.58G (r=16), 114.61G
(r=1), 114.66G (r=4) at N=128, D=1. Memory that does not move with r is not the
factors. The feasibility boundary is a pure function of members per device, identical
for all three ranks and both placements: every low-rank cell at 30 or more members per
device is over HBM, every one at 16 or fewer fits.

So what is the term, and why is the seed arm exempt?

## What moves it

Temporaries for one compiled generation, GiB:

**A. members per device** (8 prompts of 128 tokens, D=1)

| members | seed | rank 1 | rank 4 | rank 16 |
|---|---|---|---|---|
| 2 | 1.223 | 1.726 | 1.155 | 1.169 |
| 4 | 0.721 | 3.452 | 2.310 | 2.337 |
| 8 | 0.721 | 6.904 | 4.620 | 4.674 |
| 16 | 0.721 | 13.808 | 9.257 | 9.399 |

The seed arm is flat. The low-rank arms double with every doubling of the population.

**B. prompts in the batch** (4 members per device, D=1)

| prompts | seed | rank 1 | rank 4 | rank 16 |
|---|---|---|---|---|
| 2 | 0.289 | 0.864 | 0.584 | 0.612 |
| 4 | 0.433 | 1.727 | 1.159 | 1.187 |
| 8 | 0.721 | 3.452 | 2.310 | 2.337 |
| 16 | 1.296 | 6.903 | 4.611 | 4.638 |

Every arm scales with the batch. Perturbation storage, of either kind, does not depend
on how many prompts it is evaluated against, so the term is activations. At Qwen's
vocabulary the logits dominate: `(members, prompts, T-1, 151936)`, and `nll` takes a
log-softmax over them in f32.

**C. rank** (8 members per device, 8 prompts): 0.721 seed, 6.904 r=1, 4.620 r=4,
4.674 r=16. Not increasing in r. Rank 1 is the largest of the three, which is the
opposite of the frozen prediction.

**D. the seed arm made to evaluate like the low-rank arms.** `SeedRegenerated` takes a
`chunk`; `run_es.py` builds `mirrored_seed` as `SeedRegenerated(chunk=1)`, so it scans
one member at a time and holds one member's activations, which is its documented
O(|params|) guarantee. `LowRank` has no such knob, because its speed *is* the batching:
the base weight is unbatched under vmap, so every member shares one GEMM. Setting the
seed arm's chunk to its whole per-device population should therefore reproduce the
low-rank curve:

| members | seed, chunk=1 | seed, unchunked | rank 1 |
|---|---|---|---|
| 2 | 1.223 | 1.223 | 1.726 |
| 4 | 0.721 | 2.446 | 3.452 |
| 8 | 0.721 | 4.819 | 6.904 |
| 16 | 0.721 | 9.572 | 13.808 |

It does. Un-chunked, the seed arm goes from flat to linear in the population and lands
in the same band as the low-rank arms; chunked, it is flat.

## The reading E17 should have had

The memory column does not price the perturbation. It prices the evaluation: how many
members' activations are live at once. The seed arm fits everywhere because it scans
its members one at a time, and the low-rank arms do not because they evaluate the whole
per-device population in a single vmap, which is exactly where their speed advantage
comes from. Rank is irrelevant to it, and the low-rank factors, the thing the paper
calls cheap, really are cheap: they are far too small to show up here at all.

That is a trade, but not the one E17 named, and it runs the other way. The arm the
paper sells as the memory-light perturbation is the one that runs out of memory,
because it buys its speed by batching members, and batching members costs activations.
It is also not a fixed property of the two schemes: the seed arm's advantage is a
chunk setting, and it pays for it in decode bandwidth, while the low-rank arms would
have to give up the shared base GEMM to get it.

Nothing here touches the timings. The A/B ratios in F9 compare two placements of the
same arm at the same shape, so the chunking is common to both and cancels. The one
number it does reach is the cross-arm speedup E17 quoted (rank 1 under A against the
seed arm's best placement at D=8, N=32), which compares a batched evaluation against a
one-at-a-time one and is not a like-for-like comparison of perturbation schemes.
