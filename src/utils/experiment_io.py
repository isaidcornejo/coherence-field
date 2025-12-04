import os
import numpy as np
import matplotlib.pyplot as plt

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_fig_dir():
    return os.path.join(get_root_dir(), "paper", "figures", "generated")

def get_results_dir():
    return os.path.join(get_root_dir(), "results")

def save_spectrum(eigvals, filename, title=None):
    fig_dir = get_fig_dir()
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.plot(eigvals, marker='o')
    if title:
        plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=200)
    plt.close()

def save_results(results_dict, filename):
    res_dir = get_results_dir()
    os.makedirs(res_dir, exist_ok=True)
    np.savez(os.path.join(res_dir, filename), **results_dict)
