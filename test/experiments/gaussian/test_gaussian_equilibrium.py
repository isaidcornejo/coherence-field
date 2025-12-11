import numpy as np
from numpy.testing import assert_allclose
from src.experiments.gaussian.equilibrium import compute_gaussian_equilibrium


def test_gaussian_equilibrium_basic():
    """
    The Gaussian equilibrium routine must return all core diagnostic objects:
        - G         : analytic Fisher information
        - C         : empirical score covariance
        - H         : Fisher-normalized alignment operator
        - lambdas   : eigenvalues of H
        - A         : scalar alignment deviation
        - phi       : rectified amplitude
        - mu, sigma : model parameters
        - num_samples : Monte Carlo sample count

    This test ensures that the function provides the complete alignment
    diagnostic bundle requested throughout the experimental pipeline.
    """
    result = compute_gaussian_equilibrium(num_samples=50_000)

    expected_keys = {
        "G", "C", "H", "lambdas", "A", "phi",
        "mu", "sigma", "num_samples"
    }
    assert expected_keys.issubset(result.keys())


def test_gaussian_fisher_is_correct_shape():
    """
    The Fisher information matrix for a 2-parameter Gaussian (mean, variance)
    must be 2×2. Even if the values are not checked here, the structure of the
    matrix is essential for consistency with the information-geometric model.
    """
    result = compute_gaussian_equilibrium(num_samples=50_000)
    G = result["G"]

    assert G.shape == (2, 2)


def test_gaussian_equilibrium_C_approx_G():
    """
    Key property of exponential families:
        When q = p (equilibrium), the empirical covariance of the score
        approximates the Fisher information:

            C ≈ G

    With finite sampling, C will differ from G by O(1/sqrt(N)).
    A 1–3% relative tolerance is therefore appropriate for N ≈ 2×10^5.
    """
    result = compute_gaussian_equilibrium(num_samples=200_000)

    G = result["G"]
    C = result["C"]

    assert_allclose(C, G, rtol=0.03, atol=0.03)


def test_gaussian_equilibrium_A_near_zero():
    """
    For a Gaussian model, the alignment scalar

        A = Σ_i (λ_i − 1)

    must be approximately zero when q = p, because the eigenvalues λ_i of the
    alignment operator H = G^{-1}C satisfy λ_i ≈ 1 at equilibrium.

    Due to sampling noise, A is small but not exactly zero, typically
    |A| ~ 1/sqrt(N), which for N = 2×10^5 is of order 10^{-2}.
    """
    result = compute_gaussian_equilibrium(num_samples=200_000)
    A = result["A"]

    assert abs(A) < 0.05  # sampling-scale deviation


def test_gaussian_equilibrium_phi_small():
    """
    The rectified amplitude is defined as:

        φ = max{ sqrt(A), 0 }.

    At Gaussian equilibrium, A is close to zero but may be slightly positive
    due to Monte Carlo fluctuations. Therefore, φ is expected to be:
        - Nonzero (small) when A > 0
        - Exactly zero when A ≤ 0

    For N ≈ 2×10^5, φ typically lies in the range 0–0.04.
    """
    result = compute_gaussian_equilibrium(num_samples=200_000)
    phi = result["phi"]

    assert phi < 0.05


def test_gaussian_H_is_symmetric():
    """
    The alignment operator

        H = G^{-1} C

    is similar to the symmetric matrix G^{-1/2} C G^{-1/2}, and therefore must
    be symmetric up to floating-point error even when computed directly via
    matrix multiplication.

    A symmetric H ensures real eigenvalues and a well-defined spectrum for A.
    """
    result = compute_gaussian_equilibrium(num_samples=100_000)
    H = result["H"]

    assert_allclose(H, H.T, atol=1e-6)
