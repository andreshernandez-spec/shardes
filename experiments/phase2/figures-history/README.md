# Superseded phase 2 figures

`../figures/` is the current set. Everything here is what an earlier section of
`docs/03-phase2-benchmarks.md` reported at the time, kept because those sections
still cite it and a superseded number with no picture behind it is hard to check.

Nothing here is regenerated. If a figure here disagrees with `../figures/`, the
one here is the older answer, not a bug to reconcile.

| directory | run | commit | why it was superseded |
|---|---|---|---|
| `2026-08-06-prefix/` | first 8x A100 sweep, 256 configs | `a496345` and earlier | measured before the `seed_regenerated` scan fix, so its efficiency numbers describe a program that no longer ships. Its M1 also has the keying bug that plotted one facet's worth of data. |
| `2026-08-11-postfix/` | re-run after the scan fix, 256 configs | `a496345` | correct, but mixed three commits across the session, and M6 peak memory was wrong by a constant parameter-sized term. |
| `2026-08-14-consistent/` | single-commit re-run, 256 configs | `5769751` | still current for what it covers. Superseded only in scope: four strategies, no `mirrored_seed`. |

The current `../figures/` is the union of `results-consistent` and `results-qiu`,
320 results, five strategies. See `docs/03-phase2-benchmarks.md` for why combining
those two runs is legitimate when this document spends three sections warning
against stitching runs together.

`../figures-rehearsal/` is not here and is not committed. It is CPU dress-rehearsal
output, watermarked as not-a-scaling-measurement, regenerated whenever the rehearsal
step runs.
