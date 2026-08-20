# paper/

The MLSys-style paper. `main.tex` explains the skeleton's conventions in its
header comment; `sections/` holds one file per section, each opening with the
claim it must carry and where its evidence lives; `generated/` holds tables
emitted by `experiments/phase2/tb{1,2,3}.py --latex` and is never edited by
hand (`make tables`). Figures are included straight from `experiments/*/figures`,
so regenerating a figure updates the paper on the next build.

Swap the document class for the official MLSys one when the CFP lands.
