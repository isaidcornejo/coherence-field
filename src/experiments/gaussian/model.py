import numpy as np

def gaussian_sample(mu, sigma, num_samples, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=num_samples)
