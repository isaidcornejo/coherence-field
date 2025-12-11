import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import (
    get_root_dir,
    get_fig_dirs,
    get_results_dir,
)


# ---------------------------------------------------------
# Image utilities
# ---------------------------------------------------------

def ensure_min_resolution(fig, min_pixels=1200):
    """
    Ensure the figure width is at least min_pixels by scaling uniformly.
    """
    current_px = fig.get_figwidth() * fig.get_dpi()
    if current_px < min_pixels:
        scale = min_pixels / current_px
        fig.set_size_inches(
            fig.get_figwidth() * scale,
            fig.get_figheight() * scale
        )


def save_spectrum(eigvals, filename, title=None, dpi=300, min_pixels=1200):
    """
    Save an eigenvalue spectrum plot into both publication directories.
    """
    fig = plt.figure(figsize=(5, 3), dpi=dpi)
    plt.plot(eigvals, marker='o')
    if title:
        plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()

    ensure_min_resolution(fig, min_pixels=min_pixels)

    for directory in get_fig_dirs():
        os.makedirs(directory, exist_ok=True)
        fig.savefig(os.path.join(directory, filename), dpi=dpi)

    plt.close(fig)


# ---------------------------------------------------------
# Results saving
# ---------------------------------------------------------

def save_results(results_dict, filename):
    """
    Save NPZ experiment outputs into 'results/' directory.

    Parameters
    ----------
    results_dict : dict
    filename     : str
    """
    out_dir = get_results_dir()
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, filename), **results_dict)
