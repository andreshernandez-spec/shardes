# paper/

The MLSys-style paper. `main.tex` explains the skeleton's conventions in its
header comment; `sections/` holds one file per section, each opening with the
claim it must carry and where its evidence lives; `generated/` holds tables
emitted by the three `experiments/phase2/tb*.py` scripts (tb3.py writes
both the systems ablations and the appendix validation checklist) plus
`experiments/countdown/analysis_e15.py`, `tb5_e17.py`, and `tb6_e16.py`,
and is never edited
by hand (`make tables`). Figures are included straight from
`experiments/*/figures`, so regenerating a figure updates the paper on the
next build.

Swap the document class for the official MLSys one when the CFP lands.
