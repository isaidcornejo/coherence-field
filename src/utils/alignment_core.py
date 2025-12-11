import numpy as np


# ============================================================================
# Internal utility: stable inverse square root via eigendecomposition
# ============================================================================
def _inverse_sqrt(G, eps=1e-12):
    """
    Compute the inverse square root G^{-1/2} of a symmetric positive-definite
    matrix using an eigenvalue decomposition.

    This method ensures:
        - exact symmetry,
        - numerical stability,
        - correct behaviour when C = G → H = I exactly.

    Parameters
    ----------
    G : np.ndarray
        Symmetric positive-definite matrix.
    eps : float
        Minimum allowed eigenvalue to avoid numerical instabilities.

    Returns
    -------
    np.ndarray
        The matrix G^{-1/2}, symmetric by construction.
    """

    # Enforce symmetry explicitly
    G = 0.5 * (G + G.T)

    # Eigen decomposition G = U Λ U^T
    eigvals, U = np.linalg.eigh(G)

    # Regularize eigenvalues
    eigvals = np.maximum(eigvals, eps)

    # Build Λ^{-1/2}
    inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
    Gm12 = U @ np.diag(inv_sqrt_vals) @ U.T

    return Gm12


# ============================================================================
# Alignment operator H = G^{-1/2} C G^{-1/2}
# ============================================================================
def compute_alignment_operator(G, C, eps=1e-12):
    """
    Compute the Fisher-normalized alignment operator:

        H = G^{-1/2} C G^{-1/2}

    Both G and C are symmetrized to eliminate numerical drift.

    Parameters
    ----------
    G : np.ndarray
        Fisher information matrix.
    C : np.ndarray
        Empirical score covariance.
    eps : float
        Regularization parameter used in eigenvalue flooring.

    Returns
    -------
    np.ndarray
        The symmetric alignment operator H.
    """

    # Symmetrize inputs
    G = 0.5 * (G + G.T)
    C = 0.5 * (C + C.T)

    # Compute inverse square root
    Gm12 = _inverse_sqrt(G, eps)

    # Construct alignment operator
    H = Gm12 @ C @ Gm12

    # Final symmetric projection
    return 0.5 * (H + H.T)


# ============================================================================
# Alignment scalar A = Σ (λ_i - 1)
# ============================================================================
def alignment_scalar_numpy(G, C):
    """
    Compute the scalar alignment diagnostic using eigenvalues of H:

        A = Σ (λ_i - 1)

    where λ_i are the eigenvalues of:

        H = G^{-1/2} C G^{-1/2}

    Parameters
    ----------
    G, C : np.ndarray
        Fisher matrix and empirical covariance.

    Returns
    -------
    tuple:
        A : float
            Alignment deviation from equilibrium.
        eigvals : np.ndarray
            Eigenvalues of H.
    """

    H = compute_alignment_operator(G, C)
    eigvals = np.linalg.eigvalsh(H)

    A = float(np.sum(eigvals - 1.0))
    return A, eigvals


# ============================================================================
# Rectified amplitude φ
# ============================================================================
def compute_phi(A):
    """
    Compute the rectified excess-alignment amplitude:

        φ = sqrt(A)  if A > 0
        φ = 0        otherwise

    Parameters
    ----------
    A : float
        Scalar alignment deviation.

    Returns
    -------
    float
        Non-negative amplitude.
    """

    return np.sqrt(A) if A > 0 else 0.0
