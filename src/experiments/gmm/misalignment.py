import numpy as np
from .model import gmm_sample
from .score import gmm_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)

def compute_gmm_misalignment(
    mu1_model=0.0,
    mu2_model=4.0,
    sigma_model=1.0,
    w_model=0.5,
    mu1_data=0.0,
    mu2_data=5.0,
    sigma_data=1.0,
    w_data=0.7,
    num_samples=200_000,
    seed=777,
):
    x_model = gmm_sample(mu1_model, mu2_model, sigma_model, w_model,
                         num_samples, seed=seed)
    V_model = gmm_scores(x_model, mu1_model, mu2_model, sigma_model, w_model)
    G = (V_model @ V_model.T) / float(num_samples)

    x_data = gmm_sample(mu1_data, mu2_data, sigma_data, w_data,
                        num_samples, seed=seed + 1)
    V_data = gmm_scores(x_data, mu1_model, mu2_model, sigma_model, w_model)
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
        "mu1_model": mu1_model,
        "mu2_model": mu2_model,
        "sigma_model": sigma_model,
        "w_model": w_model,
        "mu1_data": mu1_data,
        "mu2_data": mu2_data,
        "sigma_data": sigma_data,
        "w_data": w_data,
        "num_samples": num_samples,
    }
