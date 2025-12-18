# ============================================================
# Coherence Field — Makefile
# Full testing, figure generation, and paper compilation.
# ============================================================

PYTHON  = python
PYTEST  = python -m pytest -q
LATEXMK = latexmk -pdf

# ------------------------------------------------------------
# Manuscript
# ------------------------------------------------------------

PAPER_DIR  = paper
PAPER_MAIN = scalar-diagnostic-empirical-alignment.tex


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
# Paper compilation (REVTeX canonical)
# ------------------------------------------------------------

paper:
	cd $(PAPER_DIR) && $(LATEXMK) $(PAPER_MAIN)


# ------------------------------------------------------------
# Cleanup LaTeX auxiliary files
# ------------------------------------------------------------

paper-clean:
	cd $(PAPER_DIR) && $(LATEXMK) -C


# ------------------------------------------------------------
# Cleanup utilities (cross-platform)
# ------------------------------------------------------------

clean:
	$(PYTHON) -c "import os, shutil; \
	[shutil.rmtree(os.path.join(r, d), ignore_errors=True) \
	for r, ds, _ in os.walk('.', topdown=False) for d in ds if d=='__pycache__']; \
	shutil.rmtree('results', ignore_errors=True)"

clean-figures:
	$(PYTHON) -c "import shutil; \
	shutil.rmtree('paper/figures/generated', ignore_errors=True)"


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
