import numpy as np
from numpy.testing import assert_allclose

from src.utils.matrix_utils import (
    enforce_symmetric,
    safe_inverse,
    fractional_power,
)


def test_enforce_symmetric_basic():
    """
    The enforce_symmetric() utility must return a symmetric matrix obtained by:

        S = (M + M.T) / 2.

    This operation is fundamental in spectral methods, Fisher geometry, and
    alignment operators because numerical pipelines may introduce asymmetry
    through floating-point drift or non-symmetric intermediate operations.

    This test confirms:
        • S is symmetric,
        • off-diagonal averaging is performed correctly.
    """
    M = np.array([[1.0, 2.0],
                  [0.0, 4.0]])

    S = enforce_symmetric(M)

    assert_allclose(S, S.T)
    assert_allclose(S, np.array([[1.0, 1.0],
                                 [1.0, 4.0]]))


def test_safe_inverse_regularizes_singular_matrix():
    """
    safe_inverse(M) must remain numerically stable even when M is singular.

    The function performs a Tikhonov-style regularization:

        M_reg = M + eps * I

    ensuring positive definiteness and invertibility.

    This test verifies that:
        • no exception is thrown,
        • the resulting inverse contains finite values.
    """
    M = np.array([[1.0, 0.0],
                  [0.0, 0.0]])  # singular

    inv = safe_inverse(M, eps=1e-3)

    assert np.isfinite(inv).all()


def test_safe_inverse_symmetry():
    """
    The inverse of a symmetric positive-definite matrix must itself be symmetric.
    safe_inverse() enforces this property explicitly to avoid floating-point
    asymmetry that would corrupt downstream eigenvalue computations.

    This test ensures the returned matrix is symmetric up to numerical precision.
    """
    M = np.array([[2.0, 0.1],
                  [0.2, 1.0]])

    inv = safe_inverse(M)

    assert_allclose(inv, inv.T, atol=1e-8)


def test_fractional_power_identity():
    """
    For any real exponent α, the fractional power of the identity satisfies:

        I^α = I.

    Since all eigenvalues are 1, the spectral decomposition remains constant.
    Fractional power logic must therefore return the identity exactly.

    This test checks a range of exponents, including:
        • negative powers,
        • fractional powers,
        • integer powers.
    """
    I = np.eye(3)

    for alpha in [-1, -0.5, 0.5, 1, 2]:
        P = fractional_power(I, alpha)
        assert_allclose(P, I)


def test_fractional_power_positive_definite():
    """
    For a diagonal positive-definite matrix with eigenvalues λ_i, the fractional
    matrix power must satisfy:

        M^α = diag( λ_1^α, λ_2^α, ... )

    This test verifies correctness using a square-root operation on a simple
    diagonal matrix with known analytic results.
    """
    M = np.array([[4.0, 0.0],
                  [0.0, 9.0]])  # PD diagonal

    P = fractional_power(M, 0.5)

    assert_allclose(P, np.array([[2.0, 0.0],
                                 [0.0, 3.0]]))


def test_fractional_power_is_symmetric():
    """
    Fractional powers of symmetric positive-definite matrices must remain
    symmetric. The implementation uses eigen-decomposition:

        M = Q Λ Qᵀ
        M^α = Q Λ^α Qᵀ

    which preserves symmetry exactly in theory and up to numerical tolerance
    in practice.

    This test confirms symmetry of the resulting matrix.
    """
    M = np.array([[3.0, 1.0],
                  [1.0, 2.0]])

    P = fractional_power(M, 0.5)

    assert_allclose(P, P.T, atol=1e-8)
