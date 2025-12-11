import numpy as np
from .model import laplace_sample
from .score import laplace_scores
from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def compute_laplace_equilibrium(
    num_samples: int = 200_000,
    mu: float = 0.0,
    b: float = 1.0,
    seed: int = 111,
):
    """
    Compute the Fisher-equilibrium diagnostic for the univariate Laplace model:

        p(x | μ, b) = (1 / (2b)) * exp(-|x - μ| / b)

    Unlike the Gaussian, the Laplace distribution *is not* a minimal exponential
    family. Therefore, even under exact equilibrium (q = p), the empirical score
    covariance C does **not** exactly match the Fisher matrix G.

    As a consequence:
        - The alignment operator H = G^{-1/2} C G^{-1/2} does NOT become identity.
        - The spectrum λ_i typically satisfies λ_i < 1 (suppression).
        - The scalar diagnostic A = Σ_i (λ_i - 1) is systematically negative.
        - The rectified amplitude φ = 0.

    This experiment demonstrates the *structural suppression* regime in your theory.

    Pipeline:
        1. Sample x ~ Laplace(μ, b)
        2. Compute score vectors v(x|θ)
        3. Compute analytic Fisher G
        4. Compute empirical covariance C
        5. Compute eigenvalues λ_i, A, φ

    Args:
        num_samples (int): Monte Carlo sample count.
        mu (float): Laplace location parameter.
        b (float): Laplace scale parameter (> 0).
        seed (int): RNG seed.

    Returns:
        dict containing:
            "G"        – analytic Fisher (2×2)
            "C"        – empirical covariance (2×2)
            "H"        – alignment operator
            "lambdas"  – eigenvalues of H
            "A"        – scalar deviation (A < 0)
            "phi"      – rectified amplitude (always 0 here)
            parameters – metadata for reproducibility
    """

    # ---------------------------------------------------
    # 1. Sample from Laplace distribution q=p
    # ---------------------------------------------------
    x = laplace_sample(mu, b, num_samples, seed)

    # ---------------------------------------------------
    # 2. Compute Laplace scores v = (v_μ, v_b)
    # ---------------------------------------------------
    V = laplace_scores(x, mu, b)

    # ---------------------------------------------------
    # 3. Fisher information for univariate Laplace:
    #
    #    I(μ) = 1 / b²
    #    I(b) = 2 / b²
    #
    #    Off-diagonal = 0
    # ---------------------------------------------------
    G = np.array([
        [1.0 / b**2, 0.0],
        [0.0,        2.0 / b**2],
    ])

    # ---------------------------------------------------
    # 4. Empirical score covariance C under q = p
    # ---------------------------------------------------
    C = (V @ V.T) / float(num_samples)

    # ---------------------------------------------------
    # 5. Alignment diagnostics
    # ---------------------------------------------------
    A_q, eigvals = alignment_scalar_numpy(G, C)
    phi_q = compute_phi(A_q)  # will be 0 because A < 0

    return {
        "G": G,
        "C": C,
        "H": compute_alignment_operator(G, C),
        "lambdas": eigvals,
        "A": A_q,
        "phi": phi_q,
        "mu": mu,
        "b": b,
        "num_samples": num_samples,
    }
