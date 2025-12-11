import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gmm.model import gmm_sample


def test_gmm_sample_shape():
    """
    The sampling routine must return a one-dimensional NumPy array whose length
    equals the specified number of samples. This structural guarantee is essential
    because downstream components (score computation, empirical covariance, and
    alignment diagnostics) assume a flat vector of observations drawn from the
    model distribution.
    """
    x = gmm_sample(0, 4, 1, 0.5, num_samples=5000)
    assert x.shape == (5000,)


def test_gmm_sample_reproducibility():
    """
    The sampler uses numpy.random.default_rng for randomness, which ensures that
    supplying an identical seed yields fully reproducible sample sequences. This
    determinism is critical for:
        - unit testing,
        - consistent experiment replication,
        - debugging and benchmarking of alignment behavior.
    """
    x1 = gmm_sample(0, 4, 1, 0.5, num_samples=5000, seed=10)
    x2 = gmm_sample(0, 4, 1, 0.5, num_samples=5000, seed=10)

    assert_allclose(x1, x2)


def test_gmm_sample_different_seed_produces_different_output():
    """
    Distinct seeds must produce statistically different samples. While equality
    could occur by extremely small probability, the chance of two independent
    GMM draws (N = 5000) matching exactly is effectively zero.

    This test validates proper functioning of the RNG and confirms that the
    sampler is not inadvertently deterministic when the seed differs.
    """
    x1 = gmm_sample(0, 4, 1, 0.5, num_samples=5000, seed=1)
    x2 = gmm_sample(0, 4, 1, 0.5, num_samples=5000, seed=2)

    assert not np.allclose(x1, x2)


def test_gmm_sample_mixture_proportions():
    """
    For a Gaussian Mixture Model with mixture weight w, the component membership
    of each sample is drawn independently with probability:

        P(component = 1) = w
        P(component = 2) = 1 − w

    Over many samples, the empirical frequency of points closer to μ1 than to μ2
    serves as an approximate estimator of the mixture weight.

    This test verifies that the sampler produces component proportions consistent
    with the theoretical mixing probability, up to Monte Carlo tolerance.
    """
    w = 0.3
    mu1, mu2 = -2, 5
    sigma = 1.0
    N = 200_000

    x = gmm_sample(mu1, mu2, sigma, w, num_samples=N, seed=123)

    # Approximate component label: nearest-mean heuristic (sufficient here)
    comp1_estimate = np.mean((x - mu1)**2 < (x - mu2)**2)

    # Allow small deviation due to sampling noise
    assert abs(comp1_estimate - w) < 0.02


def test_gmm_sample_mean_close_to_theory():
    """
    The theoretical mean of a two-component Gaussian mixture is:

        E[x] = w * μ1 + (1 - w) * μ2.

    For large N, the empirical mean of generated samples must converge to this
    theoretical expectation at rate O(1 / sqrt(N)).

    This test verifies consistency of first-order moments—an essential requirement
    for correct curvature estimation, alignment diagnostics, and downstream
    empirical Fisher computations.
    """
    mu1, mu2 = 0.0, 4.0
    sigma = 1.0
    w = 0.25
    N = 200_000

    x = gmm_sample(mu1, mu2, sigma, w, num_samples=N, seed=222)

    expected_mean = w * mu1 + (1 - w) * mu2
    empirical_mean = np.mean(x)

    assert abs(empirical_mean - expected_mean) < 0.05
