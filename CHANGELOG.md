# Changelog

All notable changes to this repository will be documented in this file.

The format follows *Keep a Changelog*, adapted for scientific software and reproducible research workflows.

---

## [1.0.3] – 2025-12-04

### Fixed

* Corrected figure placement issues in the manuscript (`7_experiments.tex`) under the REVTeX class:

  * Removed unsupported float specifier `H`, which is not allowed in APS/REVTeX workflows.
  * Replaced all figure environments with `htbp` to ensure compliant and stable float behavior.
  * Added `\FloatBarrier` between subsections to guarantee ordering consistency and prevent float leakage across sections.

### Changed

* Improved structural reliability of the manuscript by aligning float-handling with REVTeX best practices.
* Ensured that all experiment figures now appear within their intended subsections, preserving narrative and analytical flow.

### Notes

* These fixes improve the editorial stability of the manuscript and prevent float-placement errors during compilation or journal submission workflows (e.g., PRD/APS).
* No scientific content was modified—only layout and LaTeX infrastructure behavior.

## [1.0.2] – 2025-12-04

### Changed

* Updated `.gitignore` to properly support versioned manuscripts:

  * PDFs generated in `paper/` are now excluded to avoid noise from local TeX builds.
  * Versioned PDFs inside `paper_versions/` (including `latest/`) are explicitly **allowed** and tracked.
  * Ensures a clean research workflow: only finalized manuscript versions are versioned, while development builds remain local.

### Added

* Updated documentation in `paper_versions/` to describe the versioning policy and the purpose of the `latest/` directory.
* Clarified archival strategy for versioned PDFs and optional accompanying notes.

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
