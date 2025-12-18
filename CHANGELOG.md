# Changelog

All notable changes to this repository are documented in this file.
The format follows *Keep a Changelog*, adapted for scientific software and reproducible research workflows.

---

## [1.0.6] – 2025-12-18

### Changed

* **Simplified manuscript architecture**:

  * Removed journal-specific subdirectories `paper/revtex/` and `paper/mdpi/`.
  * The repository now maintains a **single canonical manuscript** under `paper/`, written in **REVTeX** format.
  * All figures are generated into a single directory:

    ```
    paper/figures/generated/
    ```

* **Figure and path utilities refactored**:

  * Removed all references to `revtex` and `mdpi` figure paths.
  * `get_fig_dirs()` now returns a single canonical directory, while remaining list-based for future extensibility.
  * All path resolution is normalized to POSIX style for cross-platform reproducibility (Windows / Linux / CI).

* **Tests aligned with new repository contract**:

  * Updated I/O tests to reflect the single-directory figure pipeline.
  * Removed assumptions about dual-publication workflows.
  * Corrected monkeypatch targets to reflect actual import resolution in `experiment_io`.

* **Documentation corrections**:

  * Updated docstrings, comments, and README references to remove mentions of MDPI-specific workflows.
  * Clarified that journal-specific layouts (if needed) should live in separate repositories or branches.

### Notes

* No scientific content was modified.
* This change reflects a **deliberate contraction of scope**: the repository now represents a single, clean, REVTeX-based reference implementation.
* The codebase remains extensible but avoids maintaining inactive or unused publication paths.

---

## [1.0.5] – 2025-12-10

### Added

* **Full test suite integration** across all modules:

  * Comprehensive coverage for `src/experiments/`, `src/utils/`, and figure-generation pipelines.
  * Ensures numerical stability, reproducibility, and correctness of the alignment operator, scalar diagnostic, and spectral computations.

* **Makefile** providing a unified, journal-ready reproducible workflow:

  * `make test` — run complete test suite.
  * `make figures` — regenerate all manuscript figures.
  * `make paper` — compile the REVTeX manuscript.
  * `make all` — end-to-end reproducible build (clean → test → results → figures → paper).

### Changed

* **README updated extensively** to reflect the reproducible pipeline:

  * Added instructions for running tests directly using Python:

    ```bash
    python -m pytest -q
    ```

  * Added documentation for Makefile usage.

  * Updated repository structure tree.

  * Updated citation section with Concept DOI and version-specific DOIs.

### Notes

* No scientific content was modified.
* This version formalized the repository as a fully reproducible, test-validated, journal-submission-ready codebase.

---

## [1.0.4] – 2025-12-10

### Added

* Introduced **dual-manuscript architecture** to support journal-specific submission workflows (now deprecated):

  * `paper/mdpi/` for MDPI submissions.
  * `paper/revtex/` for arXiv / APS submissions.

### Notes

* This architecture has been **superseded** in version 1.0.6 in favor of a single canonical manuscript.
* No scientific content was modified.

---

## [1.0.3] – 2025-12-04

### Fixed

* Corrected figure placement issues in the REVTeX manuscript:

  * Removed unsupported `H` float specifier.
  * Replaced figure floats with `htbp`.
  * Added `\FloatBarrier` to prevent float leakage across sections.

---

## [1.0.2] – 2025-12-04

### Changed

* Updated `.gitignore` to support versioned manuscript workflows.

---

## [1.0.1] – 2025-12-04

### Changed

* Updated README to align with manuscript’s final title:
  **“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds.”**

---

## [1.0.0] – 2025-12-04

### Added

* Initial public release of the repository.
* Full implementation of the alignment operator and scalar diagnostic.
* Complete reproducible experiment suite.
* Canonical REVTeX manuscript under `paper/`.
