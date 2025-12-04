import numpy as np
from .model import gmm_sample
from .score import gmm_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)

def compute_gmm_equilibrium(
    num_samples=200_000,
    mu1=0.0,
    mu2=4.0,
    sigma=1.0,
    w=0.5,
    seed=555,
):
    x_model = gmm_sample(mu1, mu2, sigma, w, num_samples, seed=seed)
    V_model = gmm_scores(x_model, mu1, mu2, sigma, w)
    G = (V_model @ V_model.T) / float(num_samples)

    x_data = gmm_sample(mu1, mu2, sigma, w, num_samples, seed=seed + 1)
    V_data = gmm_scores(x_data, mu1, mu2, sigma, w)
    C = (V_data @ V_data.T) / float(num_samples)

    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    return {
        "G": G,
        "C": C,
        "H": compute_alignment_operator(G, C),
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu1": mu1,
        "mu2": mu2,
        "sigma": sigma,
        "w": w,
        "num_samples": num_samples,
    }
