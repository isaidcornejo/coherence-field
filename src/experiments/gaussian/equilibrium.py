import numpy as np
from .model import gaussian_sample
from .score import gaussian_scores
from src.utils.alignment_core import compute_alignment_operator, alignment_scalar_numpy, compute_phi

def compute_gaussian_equilibrium(
    num_samples=200_000,
    mu=0.0,
    sigma=1.0,
    seed=123
):

    x = gaussian_sample(mu, sigma, num_samples, seed)
    V = gaussian_scores(x, mu, sigma)

    # Fisher matrix analytic for univariate Gaussian
    G = np.array([
        [1/sigma**2,        0],
        [0,            2/sigma**2]
    ])

    # Empirical covariance C = E_q[v v^T]
    C = (V @ V.T) / float(num_samples)

    # Alignment operator, eigenvalues, A_q
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    return {
        "G": G,
        "C": C,
        "H": compute_alignment_operator(G, C),
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu": mu,
        "sigma": sigma,
        "num_samples": num_samples
    }
