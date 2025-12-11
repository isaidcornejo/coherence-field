import numpy as np

from .model import gaussian_sample
from .score import gaussian_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def compute_gaussian_misalignment(
    mu_model: float = 0.0,
    sigma_model: float = 1.0,
    mu_data: float = 1.0,
    sigma_data: float = 1.0,
    num_samples: int = 200_000,
    seed: int = 321,
):
    """
    Compute the misalignment experiment for a univariate Gaussian model.

    In this setting, the model p(x | θ) = N(mu_model, sigma_model²) is
    evaluated on data sampled from a *different* Gaussian distribution:

        q(x) = N(mu_data, sigma_data²)

    This mismatch induces non-equilibrium geometry, producing a non-identity
    alignment operator H with eigenvalues λ_i ≠ 1, a positive scalar
    diagnostic A > 0 (typically), and a non-zero rectified alignment amplitude φ.

    Pipeline:
        1. Sample x ~ q(x).
        2. Compute the score v(x | θ) under the *model* distribution.
        3. Construct analytic Fisher matrix G for the model.
        4. Estimate empirical score covariance C under q.
        5. Compute eigenvalues λ_i of the alignment operator H.
        6. Compute scalar deviation A and amplitude φ.

    Args:
        mu_model (float):     Model mean parameter μ for p(x|θ).
        sigma_model (float):  Model σ parameter.
        mu_data (float):      Data mean parameter μ for q(x).
        sigma_data (float):   Data σ parameter.
        num_samples (int):    Monte Carlo sample count.
        seed (int):           RNG seed for reproducibility.

    Returns:
        dict: {
            "G": Fisher matrix for model p,
            "C": Empirical score covariance under q,
            "H": Alignment operator H = G^{-1/2} C G^{-1/2},
            "lambdas": Eigenvalues of H,
            "A": Scalar deviation A,
            "phi": Rectified amplitude φ,
            "mu_model": model μ,
            "sigma_model": model σ,
            "mu_data": data μ,
            "sigma_data": data σ,
            "num_samples": number of samples,
        }
    """

    # -----------------------------------------------------------
    # 1. Generate data *from q(x)* = N(mu_data, sigma_data²)
    # -----------------------------------------------------------
    x = gaussian_sample(mu_data, sigma_data, num_samples, seed)

    # -----------------------------------------------------------
    # 2. Compute scores v(x|θ) under the *model* distribution p(x|θ)
    # -----------------------------------------------------------
    V = gaussian_scores(x, mu_model, sigma_model)  # shape: (2, N)

    # -----------------------------------------------------------
    # 3. Analytic Fisher matrix for univariate Gaussian
    #
    #    G = diag(1/σ², 2/σ²) for θ = (μ, σ)
    # -----------------------------------------------------------
    G = np.array(
        [
            [1.0 / sigma_model**2, 0.0],
            [0.0, 2.0 / sigma_model**2],
        ]
    )

    # -----------------------------------------------------------
    # 4. Empirical score covariance under q:
    #        C = E_q[v v^T]
    # -----------------------------------------------------------
    C = (V @ V.T) / float(num_samples)

    # -----------------------------------------------------------
    # 5. Alignment diagnostics (eigenvalues λ_i, scalar A, amplitude φ)
    # -----------------------------------------------------------
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    # Full alignment operator H for inspection or plotting
    H = compute_alignment_operator(G, C)

    # -----------------------------------------------------------
    # 6. Output dictionary for reproducibility
    # -----------------------------------------------------------
    return {
        "G": G,
        "C": C,
        "H": H,
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu_model": mu_model,
        "sigma_model": sigma_model,
        "mu_data": mu_data,
        "sigma_data": sigma_data,
        "num_samples": num_samples,
    }
