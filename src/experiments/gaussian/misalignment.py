import numpy as np
from .model import gaussian_sample
from .score import gaussian_scores
from src.utils.alignment_core import compute_alignment_operator, alignment_scalar_numpy, compute_phi

def compute_gaussian_misalignment(
    mu_model=0.0,
    sigma_model=1.0,
    mu_data=1.0,
    sigma_data=1.0,
    num_samples=200_000,
    seed=321
):

    # data from q(x) = N(mu_data, sigma_data^2)
    x = gaussian_sample(mu_data, sigma_data, num_samples, seed)

    # scores computed with model p(x|theta) = N(mu_model, sigma_model^2)
    V = gaussian_scores(x, mu_model, sigma_model)

    # analytic Fisher for model p(x|theta)
    G = np.array([
        [1/sigma_model**2,     0],
        [0,                 2/sigma_model**2]
    ])

    # empirical covariance under q
    C = (V @ V.T) / float(num_samples)

    # alignment
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    return {
        "G": G,
        "C": C,
        "H": compute_alignment_operator(G, C),
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu_model": mu_model,
        "sigma_model": sigma_model,
        "mu_data": mu_data,
        "sigma_data": sigma_data,
        "num_samples": num_samples
    }
