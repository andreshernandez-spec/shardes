# Build the paper. `make tables` re-emits every generated table from its
# script; `make` builds the PDF. Tables are never edited by hand.
LATEX=latexmk -pdf -interaction=nonstopmode -halt-on-error

all: tables
	$(LATEX) main.tex

tables:
	cd ../experiments/phase2 && python tb1.py --latex >/dev/null \
	  && python tb2.py --latex >/dev/null && python tb3.py --latex >/dev/null

clean:
	latexmk -C main.tex
