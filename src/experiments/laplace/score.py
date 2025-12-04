import numpy as np

def laplace_scores(x, mu, b):
    v_mu = np.sign(x - mu) / b
    v_b = -1.0/b + np.abs(x - mu) / (b**2)
    return np.vstack([v_mu, v_b])   # shape (2, N)
