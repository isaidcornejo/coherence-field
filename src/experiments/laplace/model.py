import numpy as np

def laplace_sample(mu: float, b: float, num_samples: int, seed: int | None = None):
    """
    Draw samples from a univariate Laplace (double exponential) distribution:

        p(x | μ, b) = (1 / (2b)) * exp(-|x - μ| / b)

    using the inverse CDF (inverse transform sampling) method.

    Method:
        If U ~ Uniform(-1/2, 1/2), then:
            X = μ - b * sign(U) * log(1 - 2|U|)
        exactly follows Laplace(μ, b).

    Args:
        mu (float): Location parameter μ.
        b (float): Scale parameter b > 0.
        num_samples (int): Number of samples to generate.
        seed (int | None): RNG seed for deterministic sampling.

    Returns:
        np.ndarray:
            Samples of shape (num_samples,) drawn from Laplace(μ, b).

    Numerical notes:
        - Uses numpy.random.default_rng() for modern, reliable randomness.
        - This method is exact (up to floating point precision), unlike
          rejection sampling or subtracting exponentials.
        - No explicit validation is performed on b; upstream code must ensure b > 0.
    """

    rng = np.random.default_rng(seed)

    # Uniform noise in (-0.5, 0.5)
    u = rng.uniform(-0.5, 0.5, size=num_samples)

    # Inverse CDF transform for Laplace distribution
    return mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))
