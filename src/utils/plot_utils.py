import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.experiment_io import get_fig_dirs


# =========================================================
# GLOBAL SCIENTIFIC STYLE
# =========================================================

def set_global_style():
    """
    Publication-quality Matplotlib styling:
    - 300 DPI minimum
    - clean axes
    - consistent line widths and fonts
    Suitable for REVTeX, MDPI, arXiv and high-quality preprints.
    """
    plt.style.use("default")

    plt.rcParams.update({
        "figure.figsize": (5.2, 3.2),
        "figure.dpi": 300,
        "savefig.dpi": 300,

        # Fonts
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,

        # Axes formatting
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 5,

        # Legend
        "legend.frameon": False,

        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# =========================================================
# FIGURE RESOLUTION ENFORCEMENT
# =========================================================

def enforce_min_resolution(fig, min_pixels=1200):
    """
    Ensures the figure has at least a given pixel width.
    Automatically rescales figsize to satisfy:

        width_px >= min_pixels

    Args:
        fig (matplotlib.figure.Figure)
        min_pixels (int): Minimum horizontal resolution in pixels.
    """
    current_px = fig.get_figwidth() * fig.get_dpi()

    if current_px < min_pixels:
        scale = min_pixels / current_px
        fig.set_size_inches(
            fig.get_figwidth() * scale,
            fig.get_figheight() * scale
        )


# =========================================================
# SAVE TO ALL FIGURE DIRECTORIES
# =========================================================

def save_clean(fig, filename):
    """
    Save a figure to *both* REVTeX and MDPI paths
    ensuring minimum DPI and pixel standards.

    Args:
        fig: Matplotlib figure
        filename: string (e.g. "gaussian_equilibrium.png")
    """
    # Enforce minimum figure quality
    enforce_min_resolution(fig, min_pixels=1200)

    for directory in get_fig_dirs():
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory, filename)
        fig.savefig(out_path, dpi=300)

    plt.close(fig)


# =========================================================
# PROFESSIONAL PLOTS
# =========================================================

def plot_spectrum(eigvals, filename, title=None):
    """
    Plot eigenvalue spectrum with publication-ready formatting.
    Saves automatically to REVTeX + MDPI directories.
    """
    set_global_style()

    fig, ax = plt.subplots()
    x = np.arange(len(eigvals))

    ax.plot(x, eigvals, marker="o", linestyle="-")
    ax.set_xlabel("Index")
    ax.set_ylabel(r"Eigenvalue $\lambda$")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    save_clean(fig, filename)


def plot_curve(values, filename, xlabel="Step", ylabel="Value", title=None):
    """
    Plot a single curve with publication-quality style.
    """
    set_global_style()
    fig, ax = plt.subplots()

    ax.plot(values, linestyle="-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    save_clean(fig, filename)


def plot_multiple_curves(curves, labels, filename,
                         xlabel="Step", ylabel="Value", title=None):
    """
    Plot multiple aligned curves with consistent styling.
    """
    set_global_style()
    fig, ax = plt.subplots()

    for curve, label in zip(curves, labels):
        ax.plot(curve, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend()
    fig.tight_layout()

    save_clean(fig, filename)
