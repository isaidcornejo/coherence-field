import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gaussian.score import gaussian_scores


def test_gaussian_scores_shape():
    """
    The Gaussian score function must return a matrix of shape (2, N), where:

        - Row 0 contains ∂/∂μ log p(x | μ, σ)
        - Row 1 contains ∂/∂σ log p(x | μ, σ)

    This structural constraint is essential for compatibility with alignment
    diagnostics, which expect a 2×N score matrix for the two model parameters.
    """
    x = np.linspace(-1, 1, 100)
    scores = gaussian_scores(x, mu=0.0, sigma=1.0)

    assert scores.shape == (2, 100)


def test_gaussian_scores_zero_mean_case():
    """
    Analytical sanity check:

    If all samples satisfy x_i = μ, then:

        v_μ(x_i) = (x_i - μ) / σ² = 0
        v_σ(x_i) = ((x_i - μ)² - σ²) / σ³ = -σ² / σ³ = -1/σ

    This test validates that the score implementation reduces to the correct
    closed-form expressions under this degenerate configuration.
    """
    mu = 2.0
    sigma = 3.0

    x = np.full(50, mu)
    scores = gaussian_scores(x, mu, sigma)

    v_mu = scores[0]
    v_sigma = scores[1]

    assert_allclose(v_mu, 0.0)
    assert_allclose(v_sigma, -1.0 / sigma)


def test_gaussian_scores_single_point():
    """
    Direct one-point evaluation of the Gaussian score:

        v_μ  = (x − μ) / σ²
        v_σ  = ((x − μ)² − σ²) / σ³

    This test confirms that the function computes these analytical expressions
    exactly for a single scalar input.
    """
    x = np.array([5.0])
    mu = 2.0
    sigma = 2.0

    v_mu_expected = (5 - 2) / (2**2)              # 3/4
    v_sigma_expected = ((5 - 2)**2 - 4) / (2**3)  # (9 - 4) / 8 = 5/8

    scores = gaussian_scores(x, mu, sigma)

    assert_allclose(scores[0, 0], v_mu_expected)
    assert_allclose(scores[1, 0], v_sigma_expected)


def test_gaussian_scores_empirical_mean_zero_when_centered():
    """
    Fundamental score identity:
        If x ~ N(μ, σ²), then E[v_μ] = 0 and E[v_σ] = 0.

    This follows because the score of any regular parametric model has zero
    expectation under the model distribution. For the Gaussian case, this yields:

        E[(x − μ)/σ²] = 0
        E[((x − μ)² − σ²)/σ³] = 0

    We validate this numerically with a large sample drawn from N(μ, σ²) using a
    fixed RNG seed to guarantee reproducibility. Empirical means should converge
    to zero at rate O(1/√N), thus tolerances of ~0.01 are appropriate for N = 2×10⁵.
    """
    N = 200_000
    mu = 0.0
    sigma = 1.0
    rng = np.random.default_rng(123)

    x = rng.normal(mu, sigma, size=N)
    scores = gaussian_scores(x, mu, sigma)

    assert abs(scores[0].mean()) < 0.01
    assert abs(scores[1].mean()) < 0.01
