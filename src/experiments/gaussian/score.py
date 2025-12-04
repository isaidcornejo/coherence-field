import numpy as np

def gaussian_scores(x, mu, sigma):
    v_mu = (x - mu) / sigma**2
    v_sigma = ((x - mu)**2 - sigma**2) / sigma**3
    return np.vstack([v_mu, v_sigma])   # shape (2, N)
