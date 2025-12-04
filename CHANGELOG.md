# Changelog
All notable changes to this repository will be documented in this file.

The format follows *Keep a Changelog*, adapted for scientific software and reproducible research projects.

---

## [1.0.0] – 2025-12-04
### Added
- Initial public release of the **Empirical Alignment Diagnostic Repository**.
- Full implementation of the empirical alignment framework based on the alignment operator  
  **H = G⁻¹ C** and the scalar diagnostic  
  **A(θ; q) = Tr(G⁻¹ C) – D**.
- Complete reproducible experiment suite:
  - Gaussian model (equilibrium and misalignment)
  - Laplace model (structural suppression)
  - Gaussian Mixture Models (multimodal reinforcement)
  - MNIST neural network curvature analysis
- Entire experimental pipeline under `src/experiments/` with modular components:
  - Score functions  
  - Fisher metric computation  
  - Empirical covariance routines  
  - Spectral normalization tools  
  - Alignment diagnostic modules  
  - Unified figure-generation scripts
- Added `generate_figures.py` producing all figures used in the manuscript.
- Included the full LaTeX source of the paper  
  **“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds”**  
  in the `paper/` directory.
- Added `paper_versions/` archive containing:
  - Version 1: *The Coherence Field: A Measure of Statistical Alignment on Fisher Manifolds*
  - Version 2: *A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds*
  - `revision_notes.md` documenting conceptual changes between versions.
- Added `environment.yml` for complete reproducibility (Python 3.11 + scientific stack).
- Added `.gitignore` optimized for Python, Jupyter, LaTeX, and scientific workflows.
- Added `CITATION.cff` using the Zenodo **Concept DOI**  
  `10.5281/zenodo.17731563`, ensuring that citations always resolve to the latest document version.

### Notes
- This is the **first commit** of the repository, containing the complete codebase,
  experiments, manuscript, version archive, and reproducibility environment.
- Future versions of the document (v3, v4, …) will be added to `paper_versions/` and
  will update the Concept DOI automatically via Zenodo.

