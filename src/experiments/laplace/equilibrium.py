import numpy as np
from .model import laplace_sample
from .score import laplace_scores
from src.utils.alignment_core import compute_alignment_operator, alignment_scalar_numpy, compute_phi

def compute_laplace_equilibrium(
    num_samples=200_000,
    mu=0.0,
    b=1.0,
    seed=111
):

    x = laplace_sample(mu, b, num_samples, seed)
    V = laplace_scores(x, mu, b)

    # Fisher analytic for univariate Laplace
    G = np.array([
        [1/b**2,      0],
        [0,        2/b**2]
    ])

    # empirical covariance under q=p
    C = (V @ V.T) / float(num_samples)

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
        "b": b,
        "num_samples": num_samples
    }
