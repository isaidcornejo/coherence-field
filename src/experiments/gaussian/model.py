import numpy as np

def gaussian_sample(mu: float, sigma: float, num_samples: int, seed: int | None = None):
    """
    Draw samples from a univariate Gaussian distribution N(mu, sigma^2).

    This function wraps NumPy's Generator.normal for reproducibility and
    controlled randomness.

    Args:
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation (must be positive).
        num_samples (int): Number of samples to generate.
        seed (int | None): Optional random seed to ensure deterministic output.

    Returns:
        np.ndarray: A 1D array of shape (num_samples,) containing samples
                    from the Gaussian distribution.

    Notes:
        - Uses numpy.random.default_rng for modern RNG behavior.
        - If seed is provided, sampling is fully reproducible.
        - No validation is done on sigma; upstream logic is expected
          to ensure sigma > 0.

    Example:
        x = gaussian_sample(mu=0.0, sigma=1.0, num_samples=1000, seed=42)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=num_samples)
