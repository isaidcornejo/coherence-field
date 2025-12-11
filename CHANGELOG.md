# Changelog

All notable changes to this repository will be documented in this file.
The format follows *Keep a Changelog*, adapted for scientific software and reproducible research workflows.

---

## [1.0.5] – 2025-12-10

### Added

* **Full test suite integration** across all modules:

  * Comprehensive coverage for `src/experiments/`, `src/utils/`, and figure-generation pipelines.
  * Ensures numerical stability, reproducibility, and correctness of the alignment operator, scalar diagnostic, and spectral computations.

* **Makefile** providing a unified, journal-ready reproducible workflow:

  * `make test` — run complete test suite.
  * `make figures` — regenerate all manuscript figures.
  * `make paper-mdpi` — compile MDPI manuscript.
  * `make paper-revtex` — compile REVTeX manuscript.
  * `make all` — end-to-end reproducible build (clean → test → results → figures → papers).

### Changed

* **README updated extensively** to reflect the final reproducible pipeline:

  * Added instructions for running tests directly using Python:

    ```bash
    python -m pytest -q
    ```
  * Added documentation for Makefile usage.
  * Updated repository structure tree.
  * Expanded paper compilation section (manual + Makefile).
  * Updated citation section with Concept DOI and version-specific DOIs.

### Notes

* No scientific content was modified.
* This version formalizes the repository as a fully reproducible, test-validated, journal-submission-ready codebase.
* Complements the structural manuscript changes introduced in `1.0.4`.

---

## [1.0.4] – 2025-12-10

### Added

* Introduced **dual-manuscript architecture** to support journal-specific submission workflows:

  * New folder `paper/mdpi/` containing the full MDPI-compatible manuscript:

    * `scalar-diagnostic-empirical-alignment.tex`
    * MDPI-local `references.bib`
    * `Definitions/` (class, logos, MDPI infrastructure)
    * Figures located in `paper/mdpi/figures/`
  * New folder `paper/revtex/` containing the REVTeX version for arXiv/JSTAT:

    * `main.tex`
    * REVTeX-local `references.bib`
    * Figures located in `paper/revtex/figures/`

* Added **separate bibliography files** for MDPI and REVTeX, ensuring:

  * Each manuscript compiles independently.
  * MDPI submissions succeed without external-path dependencies.
  * arXiv/REVTeX submissions remain fully self-contained.

### Changed

* Updated README to describe dual-manuscript workflow:

  * Which file to compile for MDPI.
  * Which file to compile for arXiv/REVTeX.
  * Submission-safe folder isolation rules.

* Updated figure paths across both manuscripts to **remove all upward references (`../`)**, guaranteeing compatibility with MDPI’s ZIP upload system and arXiv’s path restrictions.

### Notes

* These changes ensure full compliance with:

  * MDPI's requirement to upload a single ZIP with internal references only.
  * arXiv’s requirement for flat, dependency-contained manuscripts.
* No scientific content changed—this update is entirely structural.

---

## [1.0.3] – 2025-12-04

### Fixed

* Corrected figure placement issues in the REVTeX manuscript (`7_experiments.tex`):

  * Removed unsupported `H` float specifier.
  * Replaced figure floats with `htbp`.
  * Added `\FloatBarrier` to prevent float leakage across sections.

### Changed

* Improved reliability of float handling.
* Ensured figures appear in their intended narrative order.

### Notes

* Essential for APS/JSTAT-style reviews.
* Scientific content unchanged.

---

## [1.0.2] – 2025-12-04

### Changed

* Updated `.gitignore` to support versioned manuscript workflows:

  * Excluded PDFs in `paper/`.
  * Allowed PDFs in `paper_versions/`.

### Added

* Documentation explaining version archival strategy and use of `latest/` folder.

---

## [1.0.1] – 2025-12-04

### Changed

* Updated README to align with manuscript’s final title:
  **“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds.”**

---

## [1.0.0] – 2025-12-04

### Added

* Initial public release of the **Coherence Field / Empirical Score Alignment Diagnostic** repository.
* Full implementation of alignment operator (`H = G^{-1} C`) and scalar diagnostic (`A = Tr(G^{-1} C) – D`).
* Complete reproducible experiment suite (Gaussian, Laplace, GMM, MNIST).
* Modular experimental pipeline under `src/experiments/`.
* Unified figure generation (`generate_figures.py`).
* Full LaTeX manuscript under `paper/`.
* `paper_versions/` archive with version history and revision notes.
* `environment.yml` for reproducibility.
* Optimized `.gitignore` and `CITATION.cff`.

### Notes

* This marks the **first complete release**, establishing the scientific and reproducible foundation for ongoing development.
