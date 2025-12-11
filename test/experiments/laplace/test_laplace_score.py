import numpy as np
from numpy.testing import assert_allclose

from src.experiments.laplace.score import laplace_scores


def test_laplace_scores_shape():
    """
    The Laplace score function must return a 2×N array where:

        - Row 0 contains ∂/∂μ log p(x | μ, b)
        - Row 1 contains ∂/∂b log p(x | μ, b)

    This dimensional structure is required for Fisher information estimation,
    empirical covariance computation, and alignment operator construction.
    """
    x = np.linspace(-1, 1, 100)
    V = laplace_scores(x, mu=0.0, b=1.0)

    assert V.shape == (2, 100)


def test_laplace_scores_mu_derivative_correct():
    """
    For the Laplace(μ, b) log-likelihood, the derivative with respect to μ is:

        v_μ(x) = sign(x − μ) / b

    with the convention sign(0) = 0 for numerical stability.

    This test validates the piecewise-linear structure of the μ-score:
        - negative for x < μ,
        - zero for x = μ,
        - positive for x > μ.
    """
    mu = 0.0
    b = 2.0
    x = np.array([-1.0, 0.0, 3.0])   # negative, zero, positive relative to μ

    V = laplace_scores(x, mu, b)
    v_mu = V[0]

    # sign(-1) = -1 → -1/b
    assert_allclose(v_mu[0], -1/b)

    # sign(0) = 0 → v_μ = 0
    assert_allclose(v_mu[1], 0.0)

    # sign(+3) = +1 → +1/b
    assert_allclose(v_mu[2], 1/b)


def test_laplace_scores_b_derivative_correct():
    """
    The derivative of the Laplace log-density with respect to b is:

        v_b(x) = -1/b + |x − μ| / b²

    This expression captures the fact that b controls scale rather than location.
    This test checks the formula against small manually computed examples.
    """
    mu = 1.0
    b = 2.0
    x = np.array([1.0, 3.0])   # |x - mu| = 0 and 2

    V = laplace_scores(x, mu, b)
    v_b = V[1]

    expected_0 = -1/b + 0 / b**2     # for x = mu
    expected_1 = -1/b + 2 / b**2     # for x = 3

    assert_allclose(v_b[0], expected_0)
    assert_allclose(v_b[1], expected_1)


def test_laplace_scores_mean_zero_in_equilibrium():
    """
    A fundamental identity of score functions:

        If x ~ p(x | θ), then      E[v(x; θ)] = 0.

    For Laplace(μ, b), this means:

        E[v_μ] = 0   and   E[v_b] = 0.

    This test performs a high-precision numerical verification using an exact
    inverse-CDF sampler for Laplace and a large Monte Carlo sample (N = 200k).

    Since the score integrates to zero under the true model, the empirical mean
    of both components should converge to zero at rate O(1 / sqrt(N)).
    """
    mu = 0.0
    b = 1.5
    N = 200_000

    rng = np.random.default_rng(123)

    # Exact inverse transform sampling for Laplace
    u = rng.uniform(-0.5, 0.5, size=N)
    x = mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))

    V = laplace_scores(x, mu, b)
    v_mu_mean = V[0].mean()
    v_b_mean  = V[1].mean()

    assert abs(v_mu_mean) < 0.01
    assert abs(v_b_mean) < 0.01
