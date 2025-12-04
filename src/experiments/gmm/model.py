import numpy as np

def gmm_sample(mu1, mu2, sigma, w, num_samples, seed=None):
    """
    Sample from 1D Gaussian mixture:
    p(x) = w N(mu1, sigma^2) + (1-w) N(mu2, sigma^2)
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=num_samples) < w  # True -> comp 1, False -> comp 2
    eps = rng.normal(loc=0.0, scale=sigma, size=num_samples)
    x = np.where(z, mu1 + eps, mu2 + eps)
    return x
