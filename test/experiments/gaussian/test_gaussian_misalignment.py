import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gaussian.misalignment import compute_gaussian_misalignment


def test_gaussian_misalignment_returns_expected_keys():
    """
    The Gaussian misalignment routine must return a complete set of alignment
    diagnostics, including:

        - G, C, H      : Fisher metric, empirical covariance, and alignment operator
        - lambdas      : eigenvalues of H
        - A            : scalar alignment deviation
        - phi          : rectified excess-alignment amplitude
        - mu_model,
          sigma_model  : parameters of the model distribution p(x|θ_model)
        - mu_data,
          sigma_data   : parameters of the data distribution q(x)
        - num_samples  : number of Monte Carlo samples

    This ensures full reproducibility and conformance with the alignment
    analysis pipeline used throughout the experiments.
    """
    result = compute_gaussian_misalignment(num_samples=50_000)

    expected_keys = {
        "G",
        "C",
        "H",
        "lambdas",
        "A",
        "phi",
        "mu_model",
        "sigma_model",
        "mu_data",
        "sigma_data",
        "num_samples",
    }

    assert expected_keys.issubset(result.keys())


def test_gaussian_misalignment_G_shape():
    """
    The Fisher information matrix for a Gaussian distribution with parameters
    (μ, σ) must always have shape 2×2. This structural property must hold even
    when p and q differ, since G is determined solely by the model.
    """
    result = compute_gaussian_misalignment()
    assert result["G"].shape == (2, 2)


def test_gaussian_misalignment_C_differs_from_G():
    """
    Under misalignment (q ≠ p), the empirical covariance C of the score under q
    must differ from the Fisher information G computed under p.

    This difference reflects the anisotropic curvature induced by the mismatch,
    and is the core source of reinforcement in the Gaussian alignment spectrum.

    A generous tolerance is used because the misaligned covariance can deviate
    substantially from the Fisher structure.
    """
    result = compute_gaussian_misalignment(num_samples=80_000)

    G = result["G"]
    C = result["C"]

    assert not np.allclose(C, G, rtol=0.1, atol=0.1)


def test_gaussian_misalignment_A_positive():
    """
    In Gaussian models, misalignment induces *reinforcement* along at least one
    direction of the Fisher-orthonormal basis.

    Concretely:
        - The alignment operator H = G^{-1}C acquires an outlier eigenvalue λ > 1
        - Therefore A = Σ_i (λ_i - 1) becomes strictly positive

    This behavior is a central contrast with the Laplace distribution, where
    misalignment may strengthen suppression instead of reinforcement.
    """
    result = compute_gaussian_misalignment(num_samples=80_000)

    A = result["A"]
    assert A > 0


def test_gaussian_misalignment_phi_positive():
    """
    The rectified amplitude φ is defined as:

        φ = max{ sqrt(A), 0 }.

    Since Gaussian misalignment yields A > 0, the excess-alignment amplitude
    must be strictly positive. This test verifies the correct behavior of the
    rectification operator.
    """
    result = compute_gaussian_misalignment(num_samples=80_000)

    phi = result["phi"]
    assert phi > 0


def test_gaussian_misalignment_H_symmetric():
    """
    The alignment operator H must remain symmetric up to numerical precision.
    Even though H is computed as G^{-1}C, it is similar to the symmetric matrix:

        G^{-1/2} C G^{-1/2},

    guaranteeing real eigenvalues and a valid spectral decomposition. This test
    confirms numerical stability and correctness of the implementation.
    """
    result = compute_gaussian_misalignment(num_samples=80_000)

    H = result["H"]
    assert_allclose(H, H.T, atol=1e-6)
