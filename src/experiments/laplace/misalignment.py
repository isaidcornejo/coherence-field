import numpy as np
from .model import laplace_sample
from .score import laplace_scores
from src.utils.alignment_core import compute_alignment_operator, alignment_scalar_numpy, compute_phi

def compute_laplace_misalignment(
    mu_model=0.0,
    b_model=1.0,
    mu_data=0.0,
    b_data=0.5,
    num_samples=200_000,
    seed=222
):

    # sample from q(x)
    x = laplace_sample(mu_data, b_data, num_samples, seed)

    # compute scores under model p(x|theta)
    V = laplace_scores(x, mu_model, b_model)

    # Fisher analytic for Laplace model
    G = np.array([
        [1/b_model**2,     0],
        [0,             2/b_model**2]
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
        "b_model": b_model,
        "mu_data": mu_data,
        "b_data": b_data,
        "num_samples": num_samples
    }
