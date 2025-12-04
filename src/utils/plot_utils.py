import os
import matplotlib.pyplot as plt
import numpy as np

def set_global_style():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (5, 3),
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "font.size": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

def save_clean(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def plot_spectrum(eigvals, filename, title=None, fig_dir=None):
    set_global_style()
    
    fig, ax = plt.subplots()
    
    x = np.arange(len(eigvals))
    ax.plot(x, eigvals, marker="o")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue λ")

    save_clean(fig, os.path.join(fig_dir, filename))

def plot_curve(values, filename, xlabel="step", ylabel="value", title=None, fig_dir=None):
    set_global_style()
    
    fig, ax = plt.subplots()
    ax.plot(values)

    if title:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_clean(fig, os.path.join(fig_dir, filename))

def plot_multiple_curves(curves, labels, filename, xlabel="step", ylabel="value", title=None, fig_dir=None):
    set_global_style()
    
    fig, ax = plt.subplots()

    for curve, label in zip(curves, labels):
        ax.plot(curve, label=label)

    if title:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    save_clean(fig, os.path.join(fig_dir, filename))
