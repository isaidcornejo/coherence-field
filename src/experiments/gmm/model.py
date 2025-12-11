import numpy as np

def gmm_sample(
    mu1: float,
    mu2: float,
    sigma: float,
    w: float,
    num_samples: int,
    seed: int | None = None,
):
    """
    Draw samples from a 1D Gaussian Mixture Model (GMM) with two components:

        p(x) = w * N(mu1, sigma^2)  +  (1 - w) * N(mu2, sigma^2)

    Sampling procedure:
        1. Generate Bernoulli assignments z ~ Bernoulli(w)
           - z = True   → sample from component 1
           - z = False  → sample from component 2

        2. For each assigned component, add Gaussian noise:
           x = mu_component + eps,   eps ~ N(0, sigma^2)

    Args:
        mu1 (float): Mean of the first Gaussian component.
        mu2 (float): Mean of the second Gaussian component.
        sigma (float): Shared standard deviation (assumed > 0).
        w (float): Mixture weight for component 1, with 0 ≤ w ≤ 1.
        num_samples (int): Number of samples to generate.
        seed (int | None): Optional RNG seed for reproducibility.

    Returns:
        np.ndarray:
            A vector of shape (num_samples,) containing samples drawn
            from the specified Gaussian mixture.

    Notes:
        - Uses numpy.random.default_rng, ensuring reproducible and
          high-quality random sampling.
        - No explicit validation is done on w or sigma; upstream functions
          should ensure correct values.
    """

    # RNG instance
    rng = np.random.default_rng(seed)

    # Bernoulli assignments: True = component 1, False = component 2
    z = rng.uniform(size=num_samples) < w

    # Gaussian noise for each selected component
    eps = rng.normal(loc=0.0, scale=sigma, size=num_samples)

    # Component-wise means applied using np.where
    x = np.where(z, mu1 + eps, mu2 + eps)

    return x
