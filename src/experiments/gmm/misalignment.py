import numpy as np
from .model import gmm_sample
from .score import gmm_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def compute_gmm_misalignment(
    mu1_model: float = 0.0,
    mu2_model: float = 4.0,
    sigma_model: float = 1.0,
    w_model: float = 0.5,
    mu1_data: float = 0.0,
    mu2_data: float = 5.0,
    sigma_data: float = 1.0,
    w_data: float = 0.7,
    num_samples: int = 200_000,
    seed: int = 777,
):
    """
    Compute the Gaussian Mixture Model (GMM) misalignment experiment.

    This experiment evaluates score statistics from the *model distribution*:

        p(x | θ_model) = w * N(μ1_model, σ²_model)
                        + (1 - w) * N(μ2_model, σ²_model)

    on data generated from a *different* mixture:

        q(x) = w_data * N(μ1_data, σ²_data)
             + (1 - w_data) * N(μ2_data, σ²_data)

    This mismatch produces:
        - C ≠ G
        - Alignment operator H ≠ I
        - eigenvalues λ_i not equal to 1
        - scalar diagnostic A > 0 (typically)
        - rectified amplitude φ = sqrt(A) > 0

    Steps:
        1. Sample x_model ~ p(x|θ_model) to estimate empirical Fisher G.
        2. Sample x_data ~ q(x) to estimate C.
        3. Compute alignment operator H = G^{-1/2} C G^{-1/2}.
        4. Extract spectrum λ_i, scalar A = Σ_i (λ_i - 1), and φ.

    Args:
        Parameters define the model mixture and the data mixture separately.
        num_samples controls Monte Carlo precision.

    Returns:
        dict containing:
            - G: empirical Fisher information matrix
            - C: empirical score covariance
            - H: alignment operator
            - lambdas: eigenvalues of H
            - A: scalar alignment deviation
            - phi: rectified amplitude
            - all mixture parameters for traceability
    """

    # -----------------------------------------------------------
    # 1. Empirical Fisher matrix from model distribution p(x|θ_model)
    # -----------------------------------------------------------
    x_model = gmm_sample(
        mu1_model, mu2_model, sigma_model, w_model,
        num_samples, seed=seed
    )
    V_model = gmm_scores(
        x_model, mu1_model, mu2_model, sigma_model, w_model
    )
    G = (V_model @ V_model.T) / float(num_samples)

    # -----------------------------------------------------------
    # 2. Empirical covariance from data distribution q(x)
    # -----------------------------------------------------------
    x_data = gmm_sample(
        mu1_data, mu2_data, sigma_data, w_data,
        num_samples, seed=seed + 1
    )
    V_data = gmm_scores(
        x_data, mu1_model, mu2_model, sigma_model, w_model
    )
    C = (V_data @ V_data.T) / float(num_samples)

    # -----------------------------------------------------------
    # 3. Alignment diagnostics
    # -----------------------------------------------------------
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)
    H = compute_alignment_operator(G, C)

    # -----------------------------------------------------------
    # 4. Package results
    # -----------------------------------------------------------
    return {
        "G": G,
        "C": C,
        "H": H,
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
