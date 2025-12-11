# ============================================================
# Coherence Field — Makefile
# Full testing, figure generation, and dual-paper compilation.
# ============================================================

PYTHON      = python
PYTEST      = python -m pytest -q
LATEXMK     = latexmk -pdf

# Paper main files for MDPI and REVTeX
MDPI_MAIN   = paper/mdpi/scalar-diagnostic-empirical-alignment.tex
REVTEX_MAIN = paper/revtex/scalar-diagnostic-empirical-alignment.tex


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
test:
	$(PYTEST)

test-verbose:
	pytest


# ------------------------------------------------------------
# Figures (global reproducibility pipeline)
# ------------------------------------------------------------
figures:
	$(PYTHON) -m src.generate_figures


# ------------------------------------------------------------
# Paper compilation
# ------------------------------------------------------------
paper-mdpi:
	cd paper/mdpi && $(LATEXMK) $(MDPI_MAIN)

paper-revtex:
	cd paper/revtex && $(LATEXMK) $(REVTEX_MAIN)


# Compile both versions for submission
paper: paper-mdpi paper-revtex


# Cleanup LaTeX auxiliary files for MDPI & REVTeX
paper-clean:
	cd paper/mdpi && $(LATEXMK) -C
	cd paper/revtex && $(LATEXMK) -C


# ------------------------------------------------------------
# Formatting & Linting
# ------------------------------------------------------------
format:
	black src test

lint:
	flake8 src test


# ------------------------------------------------------------
# Cleanup utilities
# ------------------------------------------------------------
clean:
	find . -name "__pycache__" -exec rm -rf {} +
	rm -rf results/*

clean-figures:
	rm -rf paper/mdpi/figures/generated/*
	rm -rf paper/revtex/figures/generated/*


deep-clean: clean clean-figures paper-clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache


# ------------------------------------------------------------
# Full pipeline (reproducible build)
# ------------------------------------------------------------
all: clean test results figures paper
	@echo "============================================================"
	@echo "   Coherence Field — Full reproducible pipeline completed"
	@echo "============================================================"
