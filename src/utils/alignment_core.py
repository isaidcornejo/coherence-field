import numpy as np
from scipy.linalg import fractional_matrix_power


def compute_alignment_operator(G, C, eps=1e-5):
    """
    Computes the Fisher-normalized alignment operator:

        H = G^{-1/2} C G^{-1/2}

    with symmetrization and regularization for stability.
    """

    # symmetrize inputs
    G = 0.5 * (G + G.T)
    C = 0.5 * (C + C.T)

    # regularization
    G_reg = G + eps * np.eye(G.shape[0])

    # compute inverse sqrt
    Gm12 = fractional_matrix_power(G_reg, -0.5)

    # aligned operator
    H = Gm12 @ C @ Gm12

    # symmetrize H
    H = 0.5 * (H + H.T)

    return H


def alignment_scalar_numpy(G, C):
    """
    Stable computation of A = Tr(G^{-1}C) - D using the H operator.
    """

    H = compute_alignment_operator(G, C)
    eigvals = np.linalg.eigvalsh(H)

    A = np.sum(eigvals - 1.0)

    return A, eigvals

def compute_phi(A):
    """
    Rectified coherence amplitude:
        phi = sqrt(A) if A > 0 else 0
    """
    return np.sqrt(A) if A > 0 else 0.0