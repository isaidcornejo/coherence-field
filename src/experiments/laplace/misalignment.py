import numpy as np
from .model import laplace_sample
from .score import laplace_scores
from src.utils.alignment_core import compute_alignment_operator, alignment_scalar_numpy, compute_phi


def compute_laplace_misalignment(
    mu_model: float = 0.0,
    b_model: float = 1.0,
    mu_data: float = 0.0,
    b_data: float = 0.5,
    num_samples: int = 200_000,
    seed: int = 222,
):
    """
    Compute the misalignment diagnostics for the Laplace distribution.

    Model distribution:
        p(x | μ_model, b_model)

    Data distribution:
        q(x | μ_data, b_data)

    In this experiment the model and data have *different* scale parameters:
        b_model ≠ b_data.

    Consequences:
        - Empirical covariance C under q(x) differs from Fisher G of p.
        - Alignment operator H deviates from the identity.
        - Eigenvalues λ_i typically show reinforcement or suppression.
        - Scalar diagnostic A = Σ_i (λ_i - 1) becomes nonzero (often > 0).
        - Rectified amplitude φ = max{√A, 0} > 0 when A > 0.

    Steps:
        1. Generate samples from q(x) = Laplace(μ_data, b_data).
        2. Compute Laplace scores v(x|θ_model).
        3. Compute analytic Fisher matrix G for p.
        4. Compute empirical covariance C under q.
        5. Compute H, λ_i, A, φ.

    Args:
        mu_model (float): Location parameter of model p.
        b_model (float): Scale parameter of model p.
        mu_data (float): Location parameter of data q.
        b_data (float): Scale parameter of data q.
        num_samples (int): Number of Monte Carlo samples.
        seed (int): Random seed.

    Returns:
        dict containing:
            - G     : analytic Fisher matrix for p
            - C     : empirical covariance under q
            - H     : alignment operator
            - lambdas : eigenvalues of H
            - A     : scalar deviation
            - phi   : rectified coherence amplitude
            - parameters for reproducibility
    """

    # --------------------------------------------------------
    # 1. Sample from data distribution q(x | μ_data, b_data)
    # --------------------------------------------------------
    x = laplace_sample(mu_data, b_data, num_samples, seed)

    # --------------------------------------------------------
    # 2. Compute score vectors under *model* parameters p(x|θ_model)
    # --------------------------------------------------------
    V = laplace_scores(x, mu_model, b_model)

    # --------------------------------------------------------
    # 3. Analytic Fisher information for Laplace model p(x|θ)
    #
    #     I(μ) = 1 / b^2
    #     I(b) = 2 / b^2
    #
    # Off-diagonal elements vanish.
    # --------------------------------------------------------
    G = np.array([
        [1.0 / b_model**2, 0.0],
        [0.0,               2.0 / b_model**2],
    ])

    # --------------------------------------------------------
    # 4. Empirical covariance matrix under q
    # --------------------------------------------------------
    C = (V @ V.T) / float(num_samples)

    # --------------------------------------------------------
    # 5. Alignment diagnostics
    # --------------------------------------------------------
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
        "num_samples": num_samples,
    }
