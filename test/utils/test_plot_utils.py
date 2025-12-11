import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for test environments

from matplotlib import pyplot as plt
from src.utils.plot_utils import (
    set_global_style,
    save_clean,
    plot_spectrum,
    plot_curve,
    plot_multiple_curves,
)
from src.utils.experiment_io import get_fig_dirs


# ---------------------------------------------------------
# GLOBAL STYLE TESTS
# ---------------------------------------------------------

def test_set_global_style_updates_params():
    """
    The global plotting style must modify key matplotlib rcParams to enforce
    consistent appearance across all figures generated in the project.

    This includes:
        • DPI for publication quality,
        • Grid visibility,
        • Font size for readability,
        • Default line width for clarity.

    Ensuring these values are set provides reproducibility across environments
    and consistent aesthetics in both REVTeX and MDPI manuscripts.
    """
    set_global_style()

    assert matplotlib.rcParams["figure.dpi"] == 300
    assert matplotlib.rcParams["axes.grid"] is True
    assert matplotlib.rcParams["font.size"] == 11
    assert matplotlib.rcParams["lines.linewidth"] == 2.0


# ---------------------------------------------------------
# DIRECTORY REDIRECTION FIXTURE
# ---------------------------------------------------------

def test_save_clean_saves_to_both_dirs(tmp_path, monkeypatch):
    """
    save_clean(fig, filename) must save the figure to *both* configured figure
    directories returned by get_fig_dirs().

    This dual-output behavior ensures:
        • REVTeX figures are generated for APS submissions,
        • MDPI figures are produced for open-access manuscripts,
        • internal workflows remain format-agnostic.

    The test monkeypatches get_fig_dirs() to avoid writing to the real project
    tree, guaranteeing isolation and reproducibility.
    """
    fake_dir1 = tmp_path / "revtex"
    fake_dir2 = tmp_path / "mdpi"
    fake_dirs = [str(fake_dir1), str(fake_dir2)]

    monkeypatch.setattr(
        "src.utils.plot_utils.get_fig_dirs",
        lambda: fake_dirs
    )

    fig = plt.figure()
    save_clean(fig, "test.png")

    assert (fake_dir1 / "test.png").exists()
    assert (fake_dir2 / "test.png").exists()


# ---------------------------------------------------------
# INDIVIDUAL PLOTTING FUNCTIONS
# ---------------------------------------------------------

def test_plot_spectrum_saves(tmp_path, monkeypatch):
    """
    plot_spectrum(eigvals, filename) must produce the spectrum plot and save it
    to both figure directories.

    This ensures compatibility with spectral diagnostics of alignment operators
    (eigenvalues λ_i), which are used throughout the experiments.
    """
    fake_dir1 = tmp_path / "revtex"
    fake_dir2 = tmp_path / "mdpi"
    fake_dirs = [str(fake_dir1), str(fake_dir2)]

    monkeypatch.setattr(
        "src.utils.plot_utils.get_fig_dirs",
        lambda: fake_dirs
    )

    plot_spectrum(np.array([1, 2, 3]), "spec.png")

    assert (fake_dir1 / "spec.png").exists()
    assert (fake_dir2 / "spec.png").exists()


def test_plot_curve_saves(tmp_path, monkeypatch):
    """
    plot_curve(y, filename) must generate a simple 1D curve plot and save it
    into both publication figure directories.

    Used for diagnostics such as:
        • convergence of A or φ over iterations,
        • score norm trajectories,
        • loss curves.
    """
    fake_dir1 = tmp_path / "revtex"
    fake_dir2 = tmp_path / "mdpi"
    fake_dirs = [str(fake_dir1), str(fake_dir2)]

    monkeypatch.setattr(
        "src.utils.plot_utils.get_fig_dirs",
        lambda: fake_dirs
    )

    plot_curve(np.linspace(0, 1, 10), "curve.png")

    assert (fake_dir1 / "curve.png").exists()
    assert (fake_dir2 / "curve.png").exists()


def test_plot_multiple_curves_saves(tmp_path, monkeypatch):
    """
    plot_multiple_curves(curves, labels, filename) must plot several labeled
    trajectories on the same axes and save the resulting figure into both
    output directories.

    This function is used for comparisons between:
        • multiple models,
        • different noise regimes,
        • different curvature/score metrics.

    The test verifies that saving works identically for composite plots.
    """
    fake_dir1 = tmp_path / "revtex"
    fake_dir2 = tmp_path / "mdpi"
    fake_dirs = [str(fake_dir1), str(fake_dir2)]

    monkeypatch.setattr(
        "src.utils.plot_utils.get_fig_dirs",
        lambda: fake_dirs
    )

    curves = [np.arange(10), np.arange(10) * 2]
    labels = ["a", "b"]

    plot_multiple_curves(curves, labels, "multi.png")

    assert (fake_dir1 / "multi.png").exists()
    assert (fake_dir2 / "multi.png").exists()
