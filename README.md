# Coherence Field: A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds

This repository provides the full implementation, experimental pipeline, and manuscript for **A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds**—a reparameterization‑invariant framework for quantifying empirical deviations from Fisher–Rao geometry.

It includes:

* A complete experimental pipeline (Gaussian, Laplace, GMM, MNIST MLP).
* Reproducible spectral analysis of the alignment operator.
* Automated figure generation.
* A fully structured LaTeX manuscript with versioning.
* Utility modules for alignment, spectral computations, and matrix operations.
* A unified Makefile for reproducible builds (optional).

---

## 📐 Core Idea

Statistical models often exhibit anisotropy in their empirical sensitivity: reinforcement modes, dimensional collapse, and heavy‑tailed curvature spectra. To characterize these phenomena invariantly, we define:

### Scalar diagnostic

```
A(θ; q) = Tr(G⁻¹ C) – D
```

### Rectified amplitude

```
ϕ(θ; q) = max( sqrt(A), 0 )
```

Where:

* **G** — Fisher information matrix.
* **C** — empirical score covariance under distribution `q`.
* **H = G⁻¹ C** — alignment operator.
* **λᵢ** — eigenvalues of `H`.

Identity:

```
A = Σᵢ (λᵢ − 1)
```

This yields an invariant summary of empirical reinforcement (`λ>1`), suppression (`λ<1`), and equilibrium (`λ≈1`).

---

## 📂 Repository Structure

```
coherence-field/
│
├─ paper/
│   ├─ mdpi/
│   └─ revtex/
│
├─ paper_versions/
│   ├─ latest/
│   └─ v1/
│
├─ src/
│   ├─ experiments/
│   ├─ utils/
│   └─ generate_figures.py
│
├─ test/
│   ├─ experiments/
│   └─ utils/
│
├─ CITATION.cff
├─ Makefile
├─ environment.yml
├─ LICENSE
└─ README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/isaidcornejo/coherence-field.git
cd coherence-field
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate coherence
```

---

## 🧪 Running Experiments

Each experiment directory includes:

* `model.py`
* `score.py`
* equilibrium and misalignment scripts
* a `run_*.py` orchestrator

### Gaussian

```bash
python -m src.experiments.gaussian.run_gaussian
```

### GMM

```bash
python -m src.experiments.gmm.run_gmm
```

### Laplace

```bash
python -m src.experiments.laplace.run_laplace
```

### MNIST

```bash
python -m src.experiments.mnist.run_mnist
```

---

## 🧪 Running Tests (without Makefile)

You can run the full test suite directly with Python:

```bash
python -m pytest -q
```

Or the verbose mode:

```bash
python -m pytest --maxfail=1 -vv
```

---

## 📊 Figures and Reproducibility

Generate all manuscript figures with:

```bash
python -m src.generate_figures
```

Outputs are saved to:

```
paper/*/figures/generated/
```

---

## 📝 Paper Compilation

The main LaTeX entrypoints are:

```
paper/mdpi/scalar-diagnostic-empirical-alignment.tex
paper/revtex/scalar-diagnostic-empirical-alignment.tex
```

To compile **without Makefile**:

```bash
cd paper/mdpi
latexmk -pdf scalar-diagnostic-empirical-alignment.tex

cd ../revtex
latexmk -pdf scalar-diagnostic-empirical-alignment.tex
```

---

## 🛠️ Using the Makefile (recommended)

### Run all tests

```bash
make test
```

### Generate all figures

```bash
make figures
```

### Compile MDPI version

```bash
make paper-mdpi
```

### Compile REVTeX version

```bash
make paper-revtex
```

### Full reproducible pipeline

```bash
make all
```

---

## 🔖 Citation (Updated)

### Concept DOI (permanent)

```
10.5281/zenodo.17731563
```

### Version‑specific DOIs

```
v2 — 10.5281/zenodo.17810561
v1 — 10.5281/zenodo.17731564
```

### Preferred citation

```
Isaid Cornejo,
"A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds",
Information Physics Institute, 2025.
DOI: 10.5281/zenodo.17731563
```

---

## 📄 License

MIT License.

---

## 🤝 Contributions

This is an active research project. Contributions, reproducibility audits, and extensions to additional models (e.g., VAEs, diffusion models) are welcome.

---

## 📬 Contact

**Isaid Cornejo** — Information Physics Institute
