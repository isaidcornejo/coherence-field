import numpy as np

def gaussian_scores(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Compute the score (gradient of log-likelihood) for a univariate
    Gaussian model p(x | μ, σ).

    The parameter vector is θ = (μ, σ), and the Gaussian density is:

        p(x|μ,σ) = (1 / √(2πσ²)) * exp(-(x - μ)² / (2σ²))

    Therefore, the score components are:

        ∂/∂μ log p = (x - μ) / σ²
        ∂/∂σ log p = ((x - μ)² - σ²) / σ³

    Args:
        x (np.ndarray): 1D array of samples, shape (N,).
        mu (float): Mean parameter μ.
        sigma (float): Standard deviation σ (must be > 0).
    
    Returns:
        np.ndarray:
            A 2×N array where:
              - Row 0 is v_μ(x)
              - Row 1 is v_σ(x)

            Shape: (2, N)

    Notes:
        - No validation is performed on sigma; upstream logic should
          ensure σ > 0 to avoid division by zero or undefined scores.
        - This implementation matches the analytic expressions used
          in Fisher metric derivations for univariate Gaussians.
    """

    # Score wrt μ
    v_mu = (x - mu) / sigma**2

    # Score wrt σ
    v_sigma = ((x - mu)**2 - sigma**2) / sigma**3

    # Output shape: (2, N)
    return np.vstack([v_mu, v_sigma])
