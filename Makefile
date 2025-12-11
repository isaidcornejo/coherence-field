# ============================================================
# Coherence Field — Makefile
# Full testing, figure generation, and dual-paper compilation.
# ============================================================

PYTHON      = python
PYTEST      = python -m pytest -q
LATEXMK     = latexmk -pdf

# Paper main files
MDPI_MAIN   = scalar-diagnostic-empirical-alignment.tex
REVTEX_MAIN = scalar-diagnostic-empirical-alignment.tex


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
test:
	$(PYTEST)

test-verbose:
	pytest


# ------------------------------------------------------------
# Figures
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

paper: paper-mdpi paper-revtex


# ------------------------------------------------------------
# Cleanup LaTeX auxiliary files
# ------------------------------------------------------------
paper-clean:
	cd paper/mdpi && $(LATEXMK) -C
	cd paper/revtex && $(LATEXMK) -C


# ------------------------------------------------------------
# Cleanup utilities (real multiplatform)
# ------------------------------------------------------------
clean:
	$(PYTHON) -c "import os, shutil; \
	[shutil.rmtree(os.path.join(r, d), ignore_errors=True) \
	for r, ds, _ in os.walk('.', topdown=False) for d in ds if d=='__pycache__']; \
	shutil.rmtree('results', ignore_errors=True)"

clean-figures:
	$(PYTHON) -c "import shutil; \
	shutil.rmtree('paper/mdpi/figures/generated', ignore_errors=True); \
	shutil.rmtree('paper/revtex/figures/generated', ignore_errors=True)"

deep-clean: clean clean-figures paper-clean
	$(PYTHON) -c "import shutil; \
	shutil.rmtree('.pytest_cache', ignore_errors=True); \
	shutil.rmtree('.mypy_cache', ignore_errors=True)"


# ------------------------------------------------------------
# Full reproducible pipeline
# ------------------------------------------------------------
all: clean test figures paper
	@echo "============================================================"
	@echo "   Coherence Field — Full reproducible pipeline completed"
	@echo "============================================================"
