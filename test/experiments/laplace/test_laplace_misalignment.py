import numpy as np
from numpy.testing import assert_allclose

from src.experiments.laplace.misalignment import compute_laplace_misalignment


# -------------------------------------------------------------------------
# Basic structure tests
# -------------------------------------------------------------------------

def test_laplace_misalignment_keys_present():
    """
    The Laplace misalignment diagnostic must return all core fields needed for
    alignment analysis, including:

        - G        : Fisher information under the model
        - C        : empirical score covariance under the data
        - H        : alignment operator H = G^{-1} C
        - lambdas  : eigenvalues of H
        - A        : scalar alignment deviation
        - phi      : rectified amplitude
        - mu_model, b_model : model parameters
        - mu_data,  b_data  : data-generating parameters
        - num_samples       : Monte Carlo sample count

    This test ensures structural completeness of the returned dictionary.
    """
    result = compute_laplace_misalignment(num_samples=50_000)

    expected_keys = {
        "G", "C", "H", "lambdas", "A", "phi",
        "mu_model", "b_model", "mu_data", "b_data",
        "num_samples"
    }

    assert expected_keys.issubset(result.keys())


def test_laplace_misalignment_G_structure():
    """
    The Fisher information matrix for the 1D Laplace distribution has a closed
    analytic form:

        I(μ) = 1 / b^2
        I(b) = 2 / b^2

    with all off-diagonal terms equal to zero.

    This test verifies that the implementation reproduces this known structure.
    """
    result = compute_laplace_misalignment(b_model=2.0, num_samples=10_000)

    G = result["G"]
    expected = np.array([
        [1 / 2.0**2, 0],
        [0,          2 / 2.0**2],
    ])

    assert_allclose(G, expected, atol=1e-12)


def test_laplace_misalignment_C_differs_from_G():
    """
    When the data distribution differs from the model distribution (b_data ≠ b_model),
    the empirical score covariance C must diverge from the Fisher matrix G.

    Unlike Gaussian families, the Laplace model is not a minimal exponential family,
    so even small model–data mismatches induce a pronounced structural deviation.
    """
    result = compute_laplace_misalignment(num_samples=100_000)

    G = result["G"]
    C = result["C"]

    assert not np.allclose(G, C, rtol=0.05, atol=0.05)


# -------------------------------------------------------------------------
# Alignment diagnostics — Laplace-specific behavior
# -------------------------------------------------------------------------

def test_laplace_misalignment_A_negative():
    """
    In the Laplace family, when the data distribution is *more concentrated*
    than the model (i.e., b_data < b_model), the empirical curvature is
    systematically *suppressed* relative to the model’s Fisher geometry.

    This yields:

        λ_i < 1       for both eigenvalues of H
        A = Σ_i (λ_i − 1) < 0
        φ = 0

    The default parameters in compute_laplace_misalignment instantiate exactly
    this global suppression regime.
    """
    result = compute_laplace_misalignment(num_samples=150_000)
    assert result["A"] < 0.0


def test_laplace_misalignment_phi_zero():
    """
    The rectified amplitude φ is defined as:

        φ = max{ sqrt(A), 0 }

    Since A < 0 in the Laplace suppression regime, φ must identically vanish:

        φ = 0.

    This provides a clean spectral signature distinguishing suppression regimes
    (φ = 0) from reinforcement regimes (φ > 0).
    """
    result = compute_laplace_misalignment(num_samples=150_000)
    assert result["phi"] == 0.0


def test_laplace_misalignment_H_symmetric():
    """
    Although H is computed operationally as H = G^{-1} C, the true geometric
    alignment operator is *similar* to the symmetric matrix:

        H̃ = G^{-1/2} C G^{-1/2},

    which ensures that H has a real spectrum and should remain symmetric up to
    numerical precision.

    This test checks the numerical stability and correctness of the implemented
    operator.
    """
    result = compute_laplace_misalignment(num_samples=80_000)

    H = result["H"]
    assert_allclose(H, H.T, atol=1e-6)
