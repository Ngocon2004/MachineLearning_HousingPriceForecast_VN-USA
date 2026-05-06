# Makefile for the ACM Conference Proceedings report
#
# Usage:
#   make            -> build report.pdf via xelatex + bibtex
#   make figures    -> regenerate figures from the notebooks' data
#   make clean      -> remove auxiliary build files
#   make distclean  -> also remove the produced PDF

ENGINE = xelatex
LATEXMK = latexmk
BIBTEX = bibtex
PYTHON = python3

MAIN = report
FIG_SCRIPT_1 = gen_figures.py
FIG_SCRIPT_2 = gen_topkaggle.py

.PHONY: all figures clean distclean view

all: $(MAIN).pdf

# Use latexmk for robustness; xelatex gives us native UTF-8 + Libertine
$(MAIN).pdf: $(MAIN).tex references.bib figures/metrics.json
	$(LATEXMK) -$(ENGINE) -interaction=nonstopmode -halt-on-error \
		-bibtex -shell-escape $(MAIN).tex

figures: figures/metrics.json figures/metrics_topkaggle.json

figures/metrics.json: $(FIG_SCRIPT_1)
	cd .. && $(PYTHON) report/$(FIG_SCRIPT_1)

figures/metrics_topkaggle.json: $(FIG_SCRIPT_2)
	cd .. && $(PYTHON) report/$(FIG_SCRIPT_2)

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk \
	       *.synctex.gz *.run.xml *.bcf

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf
