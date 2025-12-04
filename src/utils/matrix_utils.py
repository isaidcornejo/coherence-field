import numpy as np
from scipy.linalg import fractional_matrix_power

def enforce_symmetric(M):
    return 0.5 * (M + M.T)

def safe_inverse(M, eps=1e-8):
    M = enforce_symmetric(M)
    reg = M + eps * np.eye(M.shape[0])
    return np.linalg.inv(reg)

def fractional_power(M, alpha):
    M = enforce_symmetric(M)
    return fractional_matrix_power(M, alpha)
