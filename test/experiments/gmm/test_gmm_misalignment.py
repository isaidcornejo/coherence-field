import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gmm.misalignment import compute_gmm_misalignment


def test_gmm_misalignment_keys_present():
    """
    The misalignment diagnostic for the Gaussian Mixture Model must expose a
    complete set of fields characterizing:

        • The geometric structure under the model p(x|θ_model):
                - G          : Fisher information
                - H          : alignment operator G^{-1} C
                - lambdas    : eigenvalues of H

        • The empirical structure induced by the data distribution q(x):
                - C          : empirical score covariance
                - A          : scalar deviation Σ_i (λ_i − 1)
                - phi        : rectified excess-alignment amplitude

        • Model parameters (μ1, μ2, σ, w) used to define p
        • Data parameters (μ1, μ2, σ, w) used to define q
        • Sample size for reproducibility

    This test ensures structural completeness of the diagnostic interface.
    """
    result = compute_gmm_misalignment(num_samples=50_000)

    expected = {
        "G", "C", "H", "lambdas", "A", "phi",
        "mu1_model", "mu2_model", "sigma_model", "w_model",
        "mu1_data", "mu2_data", "sigma_data", "w_data",
        "num_samples"
    }

    assert expected.issubset(result.keys())


def test_gmm_misalignment_shapes():
    """
    All core geometric tensors (G, C, H) must be square matrices of identical
    dimension. This dimension corresponds to the number of free parameters in
    the GMM model, and shape consistency is required for spectral diagnostics,
    eigenvalue computation, and alignment analysis.
    """
    result = compute_gmm_misalignment(num_samples=30_000)

    G = result["G"]
    C = result["C"]
    H = result["H"]

    assert G.shape == C.shape == H.shape
    assert G.ndim == 2
    assert G.shape[0] == G.shape[1]


def test_gmm_misalignment_C_not_equal_G():
    """
    In misalignment (q ≠ p), the empirical score covariance C must differ from
    the Fisher information G. This discrepancy captures the deformation of
    empirical curvature relative to the intrinsic geometry of the model.

    In equilibrium, C ≈ G; under misalignment, this relationship breaks, and
    deviation should exceed the small tolerances used in equilibrium tests.
    """
    result = compute_gmm_misalignment(num_samples=80_000)

    G = result["G"]
    C = result["C"]

    assert not np.allclose(C, G, rtol=0.05, atol=0.05)


def test_gmm_misalignment_A_positive():
    """
    Gaussian Mixture Models typically exhibit *reinforcement* under parameter
    misalignment. When the data distribution q differs from the model p,
    certain curvature directions become amplified, leading to eigenvalues:

        λ_i > 1

    for at least one i.

    Consequently:
        A = Σ_i (λ_i − 1) > 0.

    This behavior sharply contrasts with the Laplace model, where misalignment
    may deepen suppression instead of producing reinforcement.
    """
    result = compute_gmm_misalignment(num_samples=100_000)
    assert result["A"] > 0


def test_gmm_misalignment_phi_positive():
    """
    The rectified amplitude is defined as:

        φ = max{ sqrt(A), 0 }

    Because GMM misalignment consistently produces reinforcement (A > 0),
    the amplitude φ must also be strictly positive.

    This confirms the correct application of the rectification operator.
    """
    result = compute_gmm_misalignment(num_samples=100_000)
    assert result["phi"] > 0


def test_gmm_misalignment_H_symmetric():
    """
    The alignment operator H is theoretically symmetric, because it is similar
    to the symmetric matrix:

        G^{-1/2} C G^{-1/2},

    even though it is computed numerically via the form H = G^{-1} C.

    This test verifies numerical symmetry up to floating-point error, ensuring
    a stable spectral decomposition and valid eigenvalue-based diagnostics.
    """
    result = compute_gmm_misalignment(num_samples=60_000)
    H = result["H"]

    assert_allclose(H, H.T, atol=1e-6)
