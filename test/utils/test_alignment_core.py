import numpy as np
from numpy.testing import assert_allclose

from src.utils.alignment_core import (
    compute_alignment_operator,
    alignment_scalar_numpy,
    compute_phi,
)


def test_alignment_identity_equilibrium():
    """
    Equilibrium sanity check.

    When the empirical score covariance equals the Fisher matrix (C = G),
    the Fisher-normalized alignment operator:

        H = G^{-1/2} C G^{-1/2}

    reduces exactly to the identity. Consequently:

        λ_i = 1   for all i
        A = Σ_i (λ_i − 1) = 0
        φ = 0

    This test verifies the fundamental identity-based behavior of the alignment
    operator under perfect equilibrium conditions.
    """
    G = np.array([[2.0, 0.0], [0.0, 3.0]])
    C = G.copy()

    H = compute_alignment_operator(G, C)
    A, eigvals = alignment_scalar_numpy(G, C)

    assert_allclose(H, np.eye(2), atol=1e-6)
    assert_allclose(eigvals, np.ones(2), atol=1e-6)
    assert abs(A) < 1e-6
    assert compute_phi(A) == 0.0


def test_alignment_reinforcement_case():
    """
    Reinforcement regime.

    If C has strictly greater curvature than G along at least one direction,
    then the normalized curvature increases:

        λ_i > 1    for some i
        A = Σ_i (λ_i − 1) > 0
        φ = sqrt(A) > 0

    This corresponds to a geometric *reinforcement* regime, characteristic of
    Gaussian and GMM misalignment scenarios.
    """
    G = np.eye(2)
    C = np.array([[5.0, 0.0], [0.0, 1.0]])

    A, eigvals = alignment_scalar_numpy(G, C)

    assert eigvals[0] > 1 or eigvals[1] > 1
    assert A > 0
    assert compute_phi(A) > 0


def test_alignment_suppression_case():
    """
    Suppression regime.

    If empirical curvature is uniformly lower than Fisher curvature:

        C = α G    with 0 < α < 1

    then:

        λ_i = α < 1
        A = Σ_i (λ_i − 1) < 0
        φ = 0

    This behavior is the hallmark of Laplace equilibrium and misalignment
    when the data distribution is more concentrated than the model.
    """
    G = np.eye(2)
    C = 0.2 * np.eye(2)

    A, eigvals = alignment_scalar_numpy(G, C)

    assert np.all(eigvals < 1)
    assert A < 0
    assert compute_phi(A) == 0.0


def test_symmetrization_stability():
    """
    Numerical symmetry enforcement.

    Even if the input matrices G and C are not symmetric — e.g., due to
    floating-point artifacts or upstream approximations — the resulting
    alignment operator must be symmetric:

        H = Hᵀ.

    The implementation performs an explicit symmetrization step, ensuring that
    eigenvalues are real and that spectral diagnostics remain well-defined.
    """
    G = np.array([[2.0, 1.0], [0.0, 3.0]])    # intentionally non-symmetric
    C = np.array([[1.0, 0.0], [2.0, 4.0]])    # intentionally non-symmetric

    H = compute_alignment_operator(G, C)
    assert_allclose(H, H.T, atol=1e-6)


def test_regularization_ensures_positive_definiteness():
    """
    Regularization stability.

    When G is singular or nearly singular, a small identity perturbation
    is added internally:

        G_reg = G + ε I

    ensuring invertibility and making the normalized operator well-defined.

    This test checks that the resulting alignment operator has nonnegative
    eigenvalues, consistent with positive semidefiniteness of G^{-1/2} C G^{-1/2}.
    """
    G = np.array([[1.0, 0.0], [0.0, 0.0]])  # singular
    C = np.eye(2)

    H = compute_alignment_operator(G, C)
    eigvals = np.linalg.eigvalsh(H)

    assert np.all(eigvals >= 0)


def test_phi_behavior():
    """
    φ (phi) is defined as the rectified square-root amplitude of A:

        φ(A) = max( sqrt(A), 0 )

    It behaves like:
        - φ = 0 for A ≤ 0  (suppression)
        - φ = sqrt(A) for A > 0  (reinforcement)

    This test checks correctness at key reference points.
    """
    assert compute_phi(-5.0) == 0.0
    assert compute_phi(0.0) == 0.0
    assert compute_phi(4.0) == 2.0
