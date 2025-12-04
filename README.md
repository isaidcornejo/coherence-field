# Coherence Field: Empirical Alignment Diagnostic on Fisher Manifolds

This repository contains the full implementation, experiments, and paper associated with the **Coherence Field** / **Empirical Alignment Scalar Diagnostic**—a reparametrization‑invariant tool for quantifying empirical deviations from Fisher–Rao geometry.

The project includes:

* A complete experimental pipeline (Gaussian, Laplace, GMM, MNIST MLP).
* Reproducible spectral analysis of the alignment operator.
* Automatic figure generation.
* A fully structured LaTeX paper.
* Utility modules for alignment, spectral computations, and matrix operations.

---

## 📐 Core Idea

Modern statistical models—especially deep neural networks—exhibit strong anisotropy in empirical sensitivity: heavy‑tailed curvature spectra, reinforcement modes, and dimensional collapse.

To characterize this behavior invariantly, we define:

**Scalar diagnostic:**

```
A(θ; q) = Tr(G⁻¹C) – D
```

**Rectified amplitude:**

```
ϕ(θ; q) = max( sqrt(A), 0 )
```

Where:

* `G` is the Fisher information matrix.
* `C` is the empirical score covariance under data distribution `q`.
* `H = G⁻¹ C` is the alignment operator.
* `λᵢ` are the eigenvalues of `H`.

With the key identity:

```
A = Σᵢ (λᵢ − 1)
```

This provides a concise, invariant summary of empirical reinforcement (`λ>1`), suppression (`λ<1`), and equilibrium (`λ≈1`).

---

## 📂 Repository Structure

```
coherence-field/
│
├─ data/                     # datasets (MNIST, synthetic)
│
├─ paper/                    # LaTeX source
│   ├─ figures/              # auto-generated figures
│   ├─ sections/             # modular LaTeX sources
│   ├─ tables/               # tables included in the paper
│   └─ coherence-field.tex   # main paper
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
│       ...
│   │       └─ run_mnist.py
│   │
│   ├─ utils/
│   │   ├─ alignment_core.py
│   │   ├─ spectral_utils.py
│   │   ├─ matrix_utils.py
│   │   ├─ plot_utils.py
│   │   └─ generate_figures.py
│   │
│   └─ __init__.py
│
├─ environment.yml           # conda environment
├─ CITATION.cff              # citation metadata
├─ LICENSE
└─ README.md                 # this document
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

* a `model.py`
* `score.py`
* equilibrium and misalignment scripts
* a `run_*.py` orchestrator

### Example: Gaussian

```bash
python -m src.experiments.gaussian.run_gaussian
```

### Example: GMM

```bash
python -m src.experiments.gmm.run_gmm
```

### Example: Laplace

```bash
python -m src.experiments.laplace.run_laplace
```

### Example: MNIST

```bash
python -m src.experiments.mnist.run_mnist
```

---

## 📊 Figures and Reproducibility

All figures included in the paper can be generated via:

```bash
python -m src.utils.generate_figures
```

Output is written to `paper/figures/`.

---

## 📝 Paper

The LaTeX source for the paper is located under `paper/`. To build:

```bash
cd paper
latexmk -pdf coherence-field.tex
```

The compiled PDF is stored as `paper/coherence-field.pdf`.

---

## 📦 Versioning of Papers (Optional)

If you wish to preserve previous versions of the scientific PDF without cluttering the main directory, create:

```
paper_versions/
    v1_coherence_field.pdf
    v2_alignment_diagnostic.pdf
    ...
```

This keeps the working `paper/` clean while retaining a history of scientific releases.

This folder is **optional** and not part of typical paper repositories, but it can be useful for large theoretical evolutions.

---

## 🔖 Citation

This repository includes a `CITATION.cff` file. GitHub will automatically generate a citation entry. The canonical citation for the paper:

```
Isaid Cornejo,
"A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds",
Information Physics Institute, 2025.
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

This is an active research project. Contributions, reproducibility checks, and extensions to other models (e.g., VAEs, diffusion models) are welcome.

---

## 📬 Contact

For questions or collaboration inquiries:
**Isaid Cornejo** – Information Physics Institute

---

## 🌟 Acknowledgements

This work integrates ideas from information geometry, high‑dimensional statistics, and modern deep learning curvature studies. All experiments are fully reproducible using standard Python scientific tooling.

---

**Thank you for exploring the Coherence Field.**
