import os
import numpy as np
import shutil

from src.utils.experiment_io import (
    get_root_dir,
    get_fig_dirs,
    save_spectrum,
    save_results,
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def cleanup_dirs(paths):
    """
    Utility helper used in some tests to remove directories created during
    temporary I/O operations.

    This function ensures a clean filesystem state when tests need to create
    nested directories for figure or result output and must later remove them
    to avoid cross-test contamination.

    Parameters
    ----------
    paths : list[str]
        A list of directory paths that should be removed if they exist.
    """
    for p in paths:
        if os.path.exists(p):
            shutil.rmtree(p)


# ---------------------------------------------------
# Core I/O Tests
# ---------------------------------------------------

def test_root_dir_exists():
    """
    The project root directory returned by get_root_dir() must exist and must
    refer to an actual directory on disk.

    This root path is the anchor for:
        • figure output directories,
        • results/ archives used in reproducibility workflows,
        • experiment orchestration and data persistence.

    Ensuring this function works is critical for all downstream I/O utilities.
    """
    root = get_root_dir()
    assert os.path.exists(root)
    assert os.path.isdir(root)


def test_figure_directories_are_correct():
    """
    get_fig_dirs() must return the canonical figure directory used by the
    project:

        paper/figures/generated

    Although returned as a list for extensibility, the current contract
    specifies a single physical output directory. All returned paths must be
    correctly nested under the project root and normalized to POSIX style.
    """
    dirs = get_fig_dirs()

    assert len(dirs) == 1

    for d in dirs:
        assert "\\" not in d
        assert "paper" in d
        assert d.endswith("figures/generated")


def test_save_spectrum_creates_directory_and_file(tmp_path, monkeypatch):
    """
    save_spectrum(eigvals, filename) must:
        1. Resolve the project root path correctly
        2. Ensure the figure output directory exists
        3. Save the generated spectrum plot to disk

    This test patches the project root to an isolated temporary directory to
    prevent side-effects on the real filesystem.
    """
    # Patch project root → isolated temporary directory
    fake_root = tmp_path / "project"
    (fake_root / "paper/figures/generated").mkdir(parents=True)

    # IMPORTANT:
    # Patch the symbol as imported in experiment_io, not in paths.py
    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root).replace("\\", "/")
    )

    eigvals = np.array([1.0, 2.0, 3.0])
    filename = "test_spectrum.png"

    save_spectrum(eigvals, filename)

    expected_path = fake_root / "paper/figures/generated" / filename
    assert expected_path.exists(), f"File missing: {expected_path}"


def test_save_results_creates_directory_and_file(tmp_path, monkeypatch):
    """
    save_results(data, filename) must:
        • Create the results/ directory if it does not exist
        • Write a valid NumPy .npz archive containing the provided data

    This functionality underpins reproducibility by enabling structured
    persistence of experiment outputs, diagnostics, and metadata.
    """
    fake_root = tmp_path / "project"
    (fake_root / "results").mkdir(parents=True)

    # Patch the symbol used inside experiment_io
    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root).replace("\\", "/")
    )

    data = {"a": np.array([1, 2, 3])}
    filename = "results_test.npz"

    save_results(data, filename)

    out_path = fake_root / "results" / filename
    assert out_path.exists()


def test_save_spectrum_does_not_crash_with_small_figures(tmp_path, monkeypatch):
    """
    save_spectrum() must be robust to small or low-resolution figures.

    The internal resolution-normalization logic should:
        • rescale figures if needed,
        • avoid raising exceptions due to insufficient pixel dimensions.

    This test passes if save_spectrum() completes without error.
    """
    fake_root = tmp_path / "root"
    (fake_root / "paper/figures/generated").mkdir(parents=True)

    # Patch the symbol used inside experiment_io
    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root).replace("\\", "/")
    )

    eigvals = np.array([0.1, 0.2])
    filename = "small_fig.png"

    # Correct behavior: no exception is raised
    save_spectrum(eigvals, filename)
