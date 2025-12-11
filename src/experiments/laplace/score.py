import numpy as np


def laplace_scores(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    """
    Compute the score (gradient of log-likelihood) for the univariate Laplace distribution:

        p(x | μ, b) = (1 / (2b)) * exp(-|x - μ| / b)

    Parameter vector: θ = (μ, b)

    The analytic score components are:

        ∂/∂μ log p(x|μ,b) = sign(x - μ) / b

        ∂/∂b log p(x|μ,b) = -1/b + |x - μ| / b²

    These expressions follow directly from differentiating the log-density:

        log p = -log(2b) - |x - μ| / b

    Properties:
        - v_μ is discontinuous at x = μ, but correct analytically.
        - E[v_μ] = 0 and E[v_b] = 0 when sampling from the model (equilibrium).
        - Output is stacked as (2×N) for compatibility with alignment computations.

    Args:
        x (np.ndarray): Input samples of shape (N,).
        mu (float): Location parameter μ of the Laplace distribution.
        b (float): Scale parameter b > 0.

    Returns:
        np.ndarray: Score matrix of shape (2, N):
            - Row 0: score w.r.t. μ
            - Row 1: score w.r.t. b
    """

    # Score with respect to μ: derivative of -|x - μ| / b
    v_mu = np.sign(x - mu) / b

    # Score with respect to b:
    #   ∂b log p = -1/b + |x - μ| / b^2
    v_b = -1.0 / b + np.abs(x - mu) / (b**2)

    return np.vstack([v_mu, v_b])  # shape: (2, N)
