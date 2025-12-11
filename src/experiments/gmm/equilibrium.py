import numpy as np

from .model import gmm_sample
from .score import gmm_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def compute_gmm_equilibrium(
    num_samples: int = 200_000,
    mu1: float = 0.0,
    mu2: float = 4.0,
    sigma: float = 1.0,
    w: float = 0.5,
    seed: int = 555,
):
    """
    Compute the Fisher-equilibrium alignment diagnostics for a
    symmetric Gaussian Mixture Model (GMM) with two components.

    The model and data distributions are identical here:
        q(x) = p(x | μ1, μ2, σ, w)

    Because the GMM is *not* an exponential family, its Fisher
    information matrix does not have a simple closed form. Therefore:

        G is estimated empirically from samples drawn from the model.
        C is computed from an independently generated dataset.

    Under equilibrium we expect:
        - C ≈ G               (up to sampling noise)
        - H ≈ I               (alignment operator close to identity)
        - eigenvalues λ_i ≈ 1
        - A ≈ 0
        - φ = 0

    Args:
        num_samples (int): Number of Monte Carlo samples.
        mu1 (float): Mean of first Gaussian component.
        mu2 (float): Mean of second Gaussian component.
        sigma (float): Shared standard deviation.
        w (float): Mixture weight for component 1. (1-w for component 2)
        seed (int): Random seed for sampling.

    Returns:
        dict with fields:
            "G": empirical Fisher metric
            "C": empirical score covariance
            "H": alignment operator G^{-1/2} C G^{-1/2}
            "lambdas": eigenvalues of H
            "A": scalar alignment diagnostic
            "phi": rectified amplitude
            "mu1", "mu2", "sigma", "w", "num_samples": metadata
    """

    # ------------------------------------------------------------------
    # 1. Sample from the model distribution p(x|θ)
    # ------------------------------------------------------------------
    x_model = gmm_sample(mu1, mu2, sigma, w, num_samples, seed=seed)

    # Compute scores under the same model p
    V_model = gmm_scores(x_model, mu1, mu2, sigma, w)

    # Empirical Fisher estimate (score variance under model)
    G = (V_model @ V_model.T) / float(num_samples)

    # ------------------------------------------------------------------
    # 2. Independent dataset from the same model for empirical curvature
    # ------------------------------------------------------------------
    x_data = gmm_sample(mu1, mu2, sigma, w, num_samples, seed=seed + 1)
    V_data = gmm_scores(x_data, mu1, mu2, sigma, w)

    # Empirical covariance under q(x) = p(x|θ)
    C = (V_data @ V_data.T) / float(num_samples)

    # ------------------------------------------------------------------
    # 3. Alignment diagnostics
    # ------------------------------------------------------------------
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    H = compute_alignment_operator(G, C)

    # ------------------------------------------------------------------
    # 4. Output results
    # ------------------------------------------------------------------
    return {
        "G": G,
        "C": C,
        "H": H,
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu1": mu1,
        "mu2": mu2,
        "sigma": sigma,
        "w": w,
        "num_samples": num_samples,
    }
