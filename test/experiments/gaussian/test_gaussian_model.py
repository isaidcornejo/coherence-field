import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gaussian.model import gaussian_sample


def test_gaussian_sample_shape():
    """
    The sampling routine must return a one-dimensional NumPy array whose length
    matches `num_samples`. This ensures consistency with downstream components
    (e.g., score computation, covariance estimation) that assume a flat vector
    of observations.
    """
    x = gaussian_sample(mu=0.0, sigma=1.0, num_samples=5000, seed=123)

    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    assert x.shape == (5000,)


def test_gaussian_sample_reproducibility():
    """
    The sampler uses numpy.random.default_rng, which guarantees deterministic
    output when a seed is supplied. Two calls with identical parameters and
    identical seeds must therefore produce bitwise-identical sample arrays.

    This is essential for experiment reproducibility, controlled benchmarks,
    and unit-test determinism.
    """
    x1 = gaussian_sample(mu=1.0, sigma=2.0, num_samples=3000, seed=999)
    x2 = gaussian_sample(mu=1.0, sigma=2.0, num_samples=3000, seed=999)

    assert_allclose(x1, x2)


def test_gaussian_sample_different_seeds_produce_different_results():
    """
    Distinct seeds should, with overwhelming probability, yield distinct random
    samples. Although equality could occur by chance, the probability is
    astronomically small for nontrivial sample sizes, making this a valid
    probabilistic test for proper RNG behavior.
    """
    x1 = gaussian_sample(mu=0.0, sigma=1.0, num_samples=5000, seed=1)
    x2 = gaussian_sample(mu=0.0, sigma=1.0, num_samples=5000, seed=2)

    assert not np.allclose(x1, x2)


def test_gaussian_sample_mean_close_to_mu():
    """
    Law of Large Numbers / Central Limit Theorem check:
    The empirical mean of N IID samples from N(mu, sigma^2) should satisfy:

        |mean_hat - mu| ≲ 3 * sigma / sqrt(N)

    which corresponds to ~99.7% confidence under Gaussian concentration bounds.

    This test verifies that the sampler produces statistically consistent
    realizations whose empirical mean converges to the true parameter.
    """
    mu = 3.5
    sigma = 1.2
    N = 20_000

    x = gaussian_sample(mu, sigma, N, seed=123)
    empirical_mean = x.mean()

    std_error = sigma / np.sqrt(N)
    assert abs(empirical_mean - mu) < 3 * std_error


def test_gaussian_sample_variance_close_to_sigma_squared():
    """
    The empirical variance of a large sample from N(mu, sigma^2) should converge
    to sigma^2 at the classical Monte Carlo rate. For N ≈ 20,000, the variance
    estimate is typically within a few percent of the true value.

    This test confirms statistical consistency of second-order moments, which is
    critical for accurate empirical Fisher and covariance computations downstream.
    """
    mu = 0.0
    sigma = 2.0
    N = 20_000

    x = gaussian_sample(mu, sigma, N, seed=456)
    empirical_var = np.var(x)

    true_var = sigma**2
    # Allow a few percent deviation due to finite-sample effects
    assert abs(empirical_var - true_var) < 0.05 * true_var
