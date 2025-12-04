# Coherence Field: A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds

This repository provides the full implementation, experimental pipeline, and manuscript for the **Coherence Field** / **A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds**—a reparameterization‑invariant framework designed to quantify empirical deviations from Fisher–Rao geometry with precision and invariance.

It includes:

* A complete experimental pipeline (Gaussian, Laplace, GMM, MNIST MLP).
* Reproducible spectral analysis of the alignment operator.
* Automated figure generation.
* A fully structured LaTeX manuscript.
* Utility modules for alignment, spectral computations, and matrix operations.

---

## 📐 Core Idea

Modern statistical models often exhibit strong anisotropy in their empirical sensitivity: heavy‑tailed curvature spectra, reinforcement modes, and dimensional collapse. These effects emerge across deep neural networks, mixture models, and high‑dimensional systems.

To characterize such phenomena invariantly, we define:

**Scalar diagnostic**

```
A(θ; q) = Tr(G⁻¹ C) – D
```

**Rectified amplitude**

```
ϕ(θ; q) = max( sqrt(A), 0 )
```

Where:

* `G` — Fisher information matrix.
* `C` — empirical score covariance under distribution `q`.
* `H = G⁻¹ C` — alignment operator.
* `λᵢ` — eigenvalues of `H`.

Key identity:

```
A = Σᵢ (λᵢ − 1)
```

This provides a compact, invariant summary of empirical reinforcement (`λ>1`), suppression (`λ<1`), and equilibrium (`λ≈1`).

---

## 📂 Repository Structure

```
coherence-field/
│
├─ data/                     # datasets (MNIST, synthetic)
│
├─ paper/                    # LaTeX source
│   ├─ figures/              # auto-generated and manual figures
│   ├─ sections/
│   ├─ tables/
│   └─ coherence-field.tex
│
├─ results/                  # saved numerical results
│
├─ src/
│   ├─ experiments/
│   │   ├─ gaussian/
│   │   ├─ gmm/
│   │   ├─ laplace/
│   │   └─ mnist/
│   │       ├─ alignment.py
│   │       ├─ model.py
│   │       └─ run_mnist.py
│   │
│   ├─ utils/
│   │   ├─ alignment_core.py
│   │   ├─ spectral_utils.py
│   │   ├─ matrix_utils.py
│   │   └─ plot_utils.py
│   │
│   └─ generate_figures.py
│
├─ environment.yml
├─ CITATION.cff
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

Each experiment directory contains:

* `model.py`
* `score.py`
* equilibrium and misalignment scripts
* a dedicated `run_*.py` orchestrator

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

All figures used in the manuscript can be reproduced by running:

```bash
python -m src.generate_figures
```

Outputs are written to `paper/figures/`.

---

## 📝 Paper

To build the LaTeX manuscript:

```bash
cd paper
latexmk -pdf coherence-field.tex
```

The final compiled file is saved as:

```
paper/coherence-field.pdf
```

---

## 🔖 Citation

A `CITATION.cff` file is included. The correct reference for the work is:

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

---

Thank you for exploring the Coherence Field.
