import numpy as np

from .model import gaussian_sample
from .score import gaussian_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def compute_gaussian_equilibrium(
    num_samples: int = 200_000,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 123,
):
    """
    Compute the Fisher-equilibrium experiment for a univariate Gaussian model.

    We consider the parametric family N(μ, σ²) with parameter vector θ = (μ, σ).
    Under exact model–data equilibrium (q = p), the empirical score covariance C
    should match the Fisher information matrix G, up to sampling noise.

    This function:
        1. Samples data x ~ N(μ, σ²).
        2. Computes score vectors v(x | θ) for each sample.
        3. Builds the analytic Fisher matrix G.
        4. Estimates the empirical score covariance C.
        5. Computes the alignment operator H, its eigenvalues λ_i,
           the scalar diagnostic A = Σ_i (λ_i - 1) and the rectified
           amplitude φ = max{√A, 0}.

    Args:
        num_samples (int): Number of Monte Carlo samples from the model.
        mu (float): Mean parameter μ of the Gaussian model.
        sigma (float): Standard deviation σ > 0 of the Gaussian model.
        seed (int): Random seed for reproducible sampling.

    Returns:
        dict: Dictionary with the following entries:
            - "G":    Analytic Fisher information matrix (2x2).
            - "C":    Empirical score covariance (2x2).
            - "H":    Alignment operator H = G^{-1/2} C G^{-1/2}.
            - "lambdas": Eigenvalues of H.
            - "A":    Scalar alignment diagnostic A.
            - "phi":  Rectified coherence amplitude φ.
            - "mu":   Mean parameter used.
            - "sigma": Standard deviation used.
            - "num_samples": Number of samples used.
    """

    # ------------------------------------------------------------------
    # 1. Sample from the Gaussian model: x ~ N(μ, σ²)
    # ------------------------------------------------------------------
    x = gaussian_sample(mu, sigma, num_samples, seed)

    # ------------------------------------------------------------------
    # 2. Compute score vectors v(x | θ) for each sample.
    #    For the univariate Gaussian with θ = (μ, σ), the score typically
    #    has dimension 2: v = (∂μ log p, ∂σ log p)^T evaluated at θ.
    #    gaussian_scores must return an array of shape (D, N).
    # ------------------------------------------------------------------
    V = gaussian_scores(x, mu, sigma)  # shape: (2, N)

    # ------------------------------------------------------------------
    # 3. Analytic Fisher matrix G for univariate Gaussian N(μ, σ²).
    #
    #    For θ = (μ, σ), the Fisher matrix is:
    #        G_μμ   = 1 / σ²
    #        G_σσ   = 2 / σ²
    #        G_μσ   = G_σμ = 0
    # ------------------------------------------------------------------
    G = np.array(
        [
            [1.0 / sigma**2, 0.0],
            [0.0, 2.0 / sigma**2],
        ]
    )

    # ------------------------------------------------------------------
    # 4. Empirical covariance of the score:
    #
    #    C = E_q[ v v^T ] ≈ (1 / N) Σ_n v_n v_n^T
    #
    #    Aquí V tiene forma (D, N), así que V @ V.T ∈ R^{D×D}.
    # ------------------------------------------------------------------
    C = (V @ V.T) / float(num_samples)

    # ------------------------------------------------------------------
    # 5. Alignment diagnostic via H = G^{-1/2} C G^{-1/2}.
    #
    #    alignment_scalar_numpy(G, C) devuelve:
    #        A   = Σ_i (λ_i - 1)
    #        λ_i = eigenvalues(H)
    #
    #    compute_phi(A) devuelve φ = max{√A, 0}.
    # ------------------------------------------------------------------
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)

    # También devolvemos H explícitamente para análisis posterior.
    H = compute_alignment_operator(G, C)

    # ------------------------------------------------------------------
    # 6. Empaquetar todo en un diccionario para trazabilidad
    #    y uso directo en notebooks/figuras.
    # ------------------------------------------------------------------
    return {
        "G": G,
        "C": C,
        "H": H,
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu": mu,
        "sigma": sigma,
        "num_samples": num_samples,
    }
