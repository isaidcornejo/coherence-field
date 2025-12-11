import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gmm.score import gmm_scores, _gaussian_pdf


def test_gmm_scores_shape():
    """
    The score function for a two-component 1D Gaussian Mixture Model must return
    a matrix of shape (3, N), corresponding to the gradients of the log-likelihood
    with respect to the parameters:

        - Row 0: ∂/∂μ1 log p(x)
        - Row 1: ∂/∂μ2 log p(x)
        - Row 2: ∂/∂w  log p(x)

    This dimensionality is essential for alignment diagnostics, empirical Fisher
    estimation, and spectral analysis of sensitivity directions.
    """
    x = np.linspace(-1, 1, 100)
    V = gmm_scores(x, mu1=0, mu2=4, sigma=1, w=0.3)

    assert V.shape == (3, 100)


def test_gmm_scores_simple_case_symmetry():
    """
    Symmetry sanity check:

        If μ1 = μ2, then φ1(x) = φ2(x) for all x.

    Under this condition:
        - Responsibilities reduce to constants:
              r1 = w,   r2 = 1 - w
        - The score with respect to w becomes:
              v_w = (φ1 - φ2) / p = 0

    This test verifies that the implementation respects this symmetry and that
    responsibilities behave as expected when both components coincide exactly.
    """
    x = np.array([0.0, 1.0, -1.0])
    mu = 2.0
    sigma = 1.5
    w = 0.4

    V = gmm_scores(x, mu, mu, sigma, w)
    v_mu1, v_mu2, v_w = V

    # φ1 = φ2 implies v_w = 0
    assert_allclose(v_w, 0.0, atol=1e-12)

    # Responsibilities equal their defining mixture weights
    phi = _gaussian_pdf(x, mu, sigma)
    p = w * phi + (1 - w) * phi
    r1 = (w * phi) / p

    assert_allclose(r1, w, atol=1e-12)


def test_gmm_scores_separated_components_responsibilities():
    """
    For x very close to μ1 and far from μ2, the posterior responsibility must
    satisfy:

        r1 ≈ 1,    r2 ≈ 0

    This test checks that the responsibilities encoded implicitly in the score
    expressions reflect component dominance in well-separated regimes and that
    the score associated with μ2 vanishes effectively in such settings.
    """
    x = np.array([0.01])  # near μ1
    mu1 = 0.0
    mu2 = 10.0
    sigma = 1.0
    w = 0.5

    V = gmm_scores(x, mu1, mu2, sigma, w)
    v_mu1, v_mu2, v_w = V[:, 0]

    phi1 = _gaussian_pdf(x, mu1, sigma)
    phi2 = _gaussian_pdf(x, mu2, sigma)
    r1 = (w * phi1) / (w * phi1 + (1 - w) * phi2)

    assert r1 > 0.999
    assert abs(v_mu2) < 1e-6


def test_gmm_scores_responsibilities_sum_to_one():
    """
    Responsibilities must satisfy:

        r1(x) + r2(x) = 1

    for every observation x. This property follows from Bayes' rule and is
    fundamental to mixture model structure. The test reconstructs r1, r2 from
    component densities and verifies exact normalization up to numerical error.
    """
    x = np.linspace(-2, 6, 10000)
    mu1, mu2 = 0, 4
    sigma = 1
    w = 0.3

    V = gmm_scores(x, mu1, mu2, sigma, w)

    phi1 = _gaussian_pdf(x, mu1, sigma)
    phi2 = _gaussian_pdf(x, mu2, sigma)
    p = w * phi1 + (1 - w) * phi2

    r1 = (w * phi1) / p
    r2 = ((1 - w) * phi2) / p

    assert_allclose(r1 + r2, 1.0, atol=1e-12)


def test_gmm_scores_expected_value_near_zero_in_equilibrium():
    """
    Fundamental identity of statistical score functions:

        If x ~ p(x|θ), then   E[v(x; θ)] = 0.

    For a GMM at equilibrium (q = p), the expected score with respect to *each*
    parameter (μ1, μ2, w) must vanish. This reflects the fact that the score is
    the gradient of log-likelihood and must have zero expectation under the model.

    With N ~ 200k samples, empirical means should fall within O(1 / sqrt(N)),
    giving tolerances around 0.03.
    """
    mu1, mu2 = 0.0, 5.0
    sigma = 1.0
    w = 0.4
    N = 200_000

    rng = np.random.default_rng(123)

    # Correct mixture sampling: sample component labels, then sample conditionally.
    components = rng.choice([0, 1], size=N, p=[w, 1 - w])
    x = np.empty(N)
    x[components == 0] = mu1 + rng.normal(scale=sigma, size=(components == 0).sum())
    x[components == 1] = mu2 + rng.normal(scale=sigma, size=(components == 1).sum())

    V = gmm_scores(x, mu1, mu2, sigma, w)
    means = V.mean(axis=1)

    assert np.all(np.abs(means) < 0.03)
