import numpy as np
from numpy.testing import assert_allclose

from src.experiments.laplace.model import laplace_sample


def test_laplace_sample_shape():
    """
    The sampling function must return a 1-dimensional NumPy array whose length
    matches the number of requested samples. This structural guarantee ensures
    compatibility with downstream components such as score computation, empirical
    covariance estimation, and alignment diagnostics.
    """
    x = laplace_sample(0.0, 1.0, num_samples=5000)
    assert x.shape == (5000,)


def test_laplace_sample_reproducibility():
    """
    The Laplace sampler uses numpy.random.default_rng, so a fixed seed must
    produce identical sample sequences. This determinism is essential for:

        • Experiment reproducibility
        • Unit testing
        • Controlled benchmarking of stochastic pipelines
    """
    x1 = laplace_sample(0.0, 1.0, 10000, seed=123)
    x2 = laplace_sample(0.0, 1.0, 10000, seed=123)

    assert_allclose(x1, x2)


def test_laplace_sample_different_seed():
    """
    Distinct seeds must yield statistically different sample streams.
    While accidental collisions are theoretically possible, the probability of
    two Laplace samples of size 5000 being identical is astronomically small.

    This test verifies proper initialization of independent RNG states.
    """
    x1 = laplace_sample(0.0, 1.0, 5000, seed=1)
    x2 = laplace_sample(0.0, 1.0, 5000, seed=2)

    assert not np.allclose(x1, x2)


def test_laplace_sample_mean_approx_mu():
    """
    The Laplace distribution has mean exactly equal to μ. For large sample sizes,
    the empirical mean must converge to μ at rate O(1 / sqrt(N)).

    With N = 200,000 samples, deviations should fall well below 0.02.
    """
    mu = 2.0
    b = 1.5
    N = 200_000

    x = laplace_sample(mu, b, N, seed=10)

    empirical_mean = np.mean(x)
    assert abs(empirical_mean - mu) < 0.02


def test_laplace_sample_abs_deviation_matches_b():
    """
    A key identity of the Laplace(μ, b) distribution is:

        E[ |X − μ| ] = b.

    This expected absolute deviation is far more statistically stable than the
    variance identity Var[X] = 2 b², especially under Monte Carlo sampling.

    For large N, the empirical mean absolute deviation should converge tightly
    around b.
    """
    mu = 0.0
    b = 2.0
    N = 200_000

    x = laplace_sample(mu, b, N, seed=20)

    empirical_absdev = np.mean(np.abs(x - mu))
    assert abs(empirical_absdev - b) < 0.03
