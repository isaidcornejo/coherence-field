import numpy as np
from scipy.linalg import fractional_matrix_power


def enforce_symmetric(M):
    """
    Return a symmetrized version of the input matrix:

        M_sym = (M + M.T) / 2

    This removes numerical asymmetries that arise from floating-point
    computations and ensures compatibility with symmetric linear algebra
    routines (e.g., eigvalsh, fractional_matrix_power).

    Args:
        M (np.ndarray): Square matrix of shape (D, D).

    Returns:
        np.ndarray: Symmetric matrix (D, D).
    """
    return 0.5 * (M + M.T)


def safe_inverse(M, eps=1e-8):
    """
    Compute a numerically stable inverse of a symmetric matrix:

        inv(M + eps * I)

    This ensures positive definiteness and avoids failures when M is
    singular or ill-conditioned.

    Args:
        M (np.ndarray): Matrix to invert (D, D).
        eps (float): Small diagonal regularization. Defaults to 1e-8.

    Returns:
        np.ndarray: Inverse of (M + eps * I).

    Notes:
        - Useful when computing G^{-1} where G may be nearly singular.
        - Ensures the output is symmetric up to numerical error.
    """
    M = enforce_symmetric(M)
    reg = M + eps * np.eye(M.shape[0])
    return np.linalg.inv(reg)


def fractional_power(M, alpha):
    """
    Compute M^alpha (fractional matrix power) for a symmetric matrix M.

    Args:
        M (np.ndarray): Symmetric positive semidefinite matrix (D, D).
        alpha (float): Exponent (e.g., -1/2, 1/2).

    Returns:
        np.ndarray: Matrix M^alpha of shape (D, D).

    Notes:
        - M is symmetrized before exponentiation.
        - fractional_matrix_power requires M to be diagonalizable with
          non-negative eigenvalues when alpha is fractional.
    """
    M = enforce_symmetric(M)
    return fractional_matrix_power(M, alpha)
