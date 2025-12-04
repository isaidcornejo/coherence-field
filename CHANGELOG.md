# Changelog

All notable changes to this repository will be documented in this file.

The format follows *Keep a Changelog*, adapted for scientific software and reproducible research workflows.

---

## [1.0.1] – 2025-12-04

### Changed

* Updated `README.md` to align with the final manuscript title:
  **“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds.”**
* Revised repository description and all textual references to maintain consistency between the README, manuscript, and project nomenclature.

## [1.0.0] – 2025-12-04

### Added

* Initial public release of the **Coherence Field / Empirical Score Alignment Diagnostic Repository**.
* Full implementation of the alignment framework based on the operator
  **H = G⁻¹ C** and the scalar diagnostic
  **A(θ; q) = Tr(G⁻¹ C) – D**.
* Complete reproducible experiment suite:

  * Gaussian model (equilibrium and misalignment)
  * Laplace model (structural suppression)
  * Gaussian Mixture Models (multimodal reinforcement)
  * MNIST neural network curvature analysis
* Modular experimental pipeline under `src/experiments/` including:

  * Score functions
  * Fisher metric computation
  * Empirical covariance modules
  * Spectral analysis utilities
  * Alignment diagnostic components
  * Unified figure-generation tools
* Added `generate_figures.py` producing all figures included in the manuscript.
* Included the full LaTeX source of the paper
  **“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds”**
  in the `paper/` directory.
* Added `paper_versions/` archive containing:

  * Version 1: *The Coherence Field: A Measure of Statistical Alignment on Fisher Manifolds*
  * Version 2: *A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds*
  * `revision_notes.md` documenting conceptual changes between versions.
* Added `environment.yml` for complete reproducibility (Python 3.11 + scientific stack).
* Added `.gitignore` optimized for Python, Jupyter, LaTeX, and scientific workflows.
* Added `CITATION.cff` using the Zenodo **Concept DOI**
  `10.5281/zenodo.17731563`, ensuring that citations always resolve to the latest document version.

### Notes

* This is the **first complete release** of the repository, containing the full codebase,
  experimental suite, manuscript, version archive, and environment specification.
* Future versions of the manuscript (v3, v4, …) will be added to `paper_versions/` and
  will continue to update the Zenodo Concept DOI automatically.
