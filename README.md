# Coherence Field: A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds

This repository provides the full implementation, experimental pipeline, and manuscript for **A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds**—a reparameterization‑invariant framework for quantifying empirical deviations from Fisher–Rao geometry.

It includes:

* A complete experimental pipeline (Gaussian, Laplace, GMM, MNIST MLP).
* Reproducible spectral analysis of the alignment operator.
* Automated figure generation.
* A fully structured LaTeX manuscript with versioning.
* Utility modules for alignment, spectral computations, and matrix operations.

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
    ├─ data/                        
    │
    ├─ paper/
    │   ├─ mdpi/                     # MDPI version
    │   │
    │   ├─ revtex/                   # RevTeX version
    │
    ├─ paper_versions/               # archived PDFs
    │
    ├─ results/
    │
    ├─ src/
    │   ├─ experiments/
    │   ├─ utils/
    │   └─ generate_figures.py
    │
    ├─ CITATION.cff
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

## 📊 Figures and Reproducibility

Generate all manuscript figures with:

```bash
python -m src.generate_figures
```

Outputs are saved to:

```
paper/figures/generated/
```

---

## 📝 Paper Compilation

The main LaTeX file to compile is:

```
paper/scalar-diagnostic-empirical-alignment.tex
```

To build the manuscript:

```bash
cd paper
latexmk -pdf scalar-diagnostic-empirical-alignment.tex
```

The compiled PDF is written to:

```
paper/scalar-diagnostic-empirical-alignment.pdf
```

---

## 🔖 Citation

A `CITATION.cff` file is included. Reference:

```
Isaid Cornejo,
"A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds",
Information Physics Institute, 2025.
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
