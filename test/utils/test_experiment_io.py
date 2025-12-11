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
    temporary I/O operations. Ensures a clean environment when tests need to
    generate multiple nested folders.

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

        • figure directories (for different LaTeX formats)
        • results/ archives used in reproducibility workflows
        • experiment orchestration

    Ensuring this function works is critical for all downstream I/O utilities.
    """
    root = get_root_dir()
    assert os.path.exists(root)
    assert os.path.isdir(root)


def test_figure_directories_are_correct():
    """
    get_fig_dirs() must return the two canonical figure directories:

        paper/revtex/figures/generated
        paper/mdpi/figures/generated

    These reflect the dual publication pipeline (APS REVTeX vs MDPI), ensuring
    that all generated figures are reproduced consistently for each manuscript.
    """
    dirs = get_fig_dirs()

    assert len(dirs) == 2
    for d in dirs:
        assert "paper" in d
        assert d.endswith("figures/generated")


def test_save_spectrum_creates_both_directories(tmp_path, monkeypatch):
    """
    save_spectrum(eigvals, filename) must:
        1. Resolve the project root path
        2. Ensure both figure directories exist
        3. Save the same plot into both locations

    This test patches the project root to a temporary directory to isolate
    filesystem side-effects and verifies that two independent copies of the
    figure are created.
    """
    # Patch project root → isolated temporary directory
    fake_root = tmp_path / "project"
    (fake_root / "paper/revtex/figures/generated").mkdir(parents=True)
    (fake_root / "paper/mdpi/figures/generated").mkdir(parents=True)

    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root)
    )

    eigvals = np.array([1.0, 2.0, 3.0])
    filename = "test_spectrum.png"

    save_spectrum(eigvals, filename)

    expected_dirs = [
        fake_root / "paper/revtex/figures/generated",
        fake_root / "paper/mdpi/figures/generated",
    ]

    for d in expected_dirs:
        f = d / filename
        assert f.exists(), f"File missing: {f}"


def test_save_results_creates_directory_and_file(tmp_path, monkeypatch):
    """
    save_results(data, filename) must:
        • Create the results/ directory if it does not exist
        • Write a valid NumPy .npz archive containing the data

    This facilitates reproducibility by enabling structured storage of
    experiment outputs, model diagnostics, and spectral summaries.
    """
    fake_root = tmp_path / "project"
    (fake_root / "results").mkdir(parents=True)

    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root)
    )

    data = {"a": np.array([1, 2, 3])}
    filename = "results_test.npz"

    save_results(data, filename)

    out_path = fake_root / "results" / filename
    assert out_path.exists()


def test_save_spectrum_does_not_crash_with_small_figures(tmp_path, monkeypatch):
    """
    The helper ensure_min_resolution() inside save_spectrum() rescales figures
    with insufficient pixel dimensions.

    This test verifies robustness:
        • no errors occur even if the figure is initially too small
        • scaling logic does not introduce exceptions

    The test passes if save_spectrum() completes without raising.
    """
    fake_root = tmp_path / "root"
    (fake_root / "paper/revtex/figures/generated").mkdir(parents=True)
    (fake_root / "paper/mdpi/figures/generated").mkdir(parents=True)

    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root)
    )

    eigvals = np.array([0.1, 0.2])
    filename = "small_fig.png"

    # If no exception is raised, the behavior is correct
    save_spectrum(eigvals, filename)
