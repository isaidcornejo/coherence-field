import numpy as np

def laplace_sample(mu, b, num_samples, seed=None):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.5, 0.5, size=num_samples)
    return mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))
