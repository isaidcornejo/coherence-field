import numpy as np
from numpy.testing import assert_allclose

from src.experiments.gmm.equilibrium import compute_gmm_equilibrium


def test_gmm_equilibrium_keys_present():
    """
    The equilibrium diagnostic for the Gaussian Mixture Model must expose a
    complete set of alignment-related quantities required throughout the
    experimental pipeline:

        - G          : Fisher information matrix under the model p(x|θ)
        - C          : empirical score covariance under q = p
        - H          : alignment operator H = G^{-1} C
        - lambdas    : eigenvalues of H
        - A          : scalar alignment deviation A = Σ_i (λ_i − 1)
        - phi        : rectified amplitude max{ sqrt(A), 0 }

        - mu1, mu2   : mixture component means
        - sigma      : shared standard deviation
        - w          : mixture weight for component 1
        - num_samples: number of Monte Carlo samples used

    This test verifies the structural correctness and completeness of the
    returned result dictionary.
    """
    result = compute_gmm_equilibrium(num_samples=50_000)

    expected_keys = {
        "G", "C", "H", "lambdas", "A", "phi",
        "mu1", "mu2", "sigma", "w", "num_samples"
    }

    assert expected_keys.issubset(result.keys())


def test_gmm_equilibrium_shapes():
    """
    The Fisher matrix G, empirical covariance C, and alignment operator H must
    all be square matrices of identical dimension. This dimension corresponds
    to the number of free parameters of the GMM model (μ1, μ2, σ, w), possibly
    reduced if structural redundancies are imposed.

    This structural check ensures compatibility with eigenvalue-based alignment
    diagnostics and prevents shape mismatches in downstream computations.
    """
    result = compute_gmm_equilibrium(num_samples=30_000)

    G = result["G"]
    C = result["C"]
    H = result["H"]

    assert G.shape == C.shape == H.shape
    assert G.ndim == 2
    assert G.shape[0] == G.shape[1]


def test_gmm_H_is_symmetric():
    """
    Although the alignment operator is computed in practice as H = G^{-1}C,
    it is *theoretically* similar to the symmetric matrix:

        G^{-1/2} C G^{-1/2},

    which guarantees real eigenvalues and symmetry up to numerical error.

    This test ensures the computed operator remains symmetric within floating-
    point tolerance, validating numerical stability and consistency with the
    information-geometric theory.
    """
    result = compute_gmm_equilibrium(num_samples=50_000)
    H = result["H"]

    assert_allclose(H, H.T, atol=1e-6)


def test_gmm_equilibrium_C_approx_G():
    """
    At equilibrium (q = p), the empirical score covariance C should approximate
    the Fisher information matrix G. For regular parametric models, the identity

        C = G

    holds exactly in expectation. Empirically, Monte Carlo variance introduces
    deviations on the order of O(1 / sqrt(N)).

    A 5% tolerance is appropriate for N ≈ 2×10^5 in a 4-parameter GMM.
    """
    result = compute_gmm_equilibrium(num_samples=200_000)

    G = result["G"]
    C = result["C"]

    assert_allclose(C, G, rtol=0.05, atol=0.05)


def test_gmm_equilibrium_A_and_phi():
    """
    At equilibrium, the alignment operator satisfies λ_i ≈ 1 for all i, giving:

        A = Σ_i (λ_i − 1) ≈ 0.

    Due to small positive/negative fluctuations in A from sampling noise, the
    rectified amplitude evaluates to:

        φ = max{ sqrt(A), 0 } = 0

    because A is typically ≤ 0 or extremely small. This test confirms that the
    scalar and rectified diagnostics behave as predicted in Gaussian mixture
    equilibrium.
    """
    result = compute_gmm_equilibrium(num_samples=150_000)

    A = result["A"]
    phi = result["phi"]

    assert abs(A) < 0.05
    assert phi == 0.0
