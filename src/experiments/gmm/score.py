import numpy as np

def _gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return norm_const * np.exp(-0.5 * z**2)

def gmm_scores(x, mu1, mu2, sigma, w, eps=1e-12):
    """
    Scores wrt parameters (mu1, mu2, w), sigma fixed.
    Returns V with shape (3, N).
    """
    x = np.asarray(x)

    phi1 = _gaussian_pdf(x, mu1, sigma)
    phi2 = _gaussian_pdf(x, mu2, sigma)

    p = w * phi1 + (1.0 - w) * phi2
    p = np.maximum(p, eps)

    r1 = (w * phi1) / p
    r2 = ((1.0 - w) * phi2) / p

    v_mu1 = r1 * (x - mu1) / (sigma**2)
    v_mu2 = r2 * (x - mu2) / (sigma**2)
    v_w   = (phi1 - phi2) / p

    return np.vstack([v_mu1, v_mu2, v_w])  # (3, N)
