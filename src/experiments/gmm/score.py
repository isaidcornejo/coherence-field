import numpy as np


def _gaussian_pdf(x, mu, sigma):
    """
    Evaluate the univariate Gaussian density N(mu, sigma^2) at the points x.

    Notes
    -----
    This function returns the pointwise probability density, not the log-density.
    It is used internally for computing mixture responsibilities and score
    components in the GMM model.
    """
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return norm_const * np.exp(-0.5 * z**2)


def gmm_scores(x, mu1, mu2, sigma, w, eps=1e-12):
    """
    Compute the score function (gradient of the log-likelihood) for a
    two-component univariate Gaussian mixture model:

        p(x) = w * N(x | mu1, sigma^2) + (1 - w) * N(x | mu2, sigma^2).

    The returned scores correspond to the gradients with respect to the
    component means (mu1, mu2) and the mixture weight w.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of sample points at which the score is evaluated.
    mu1 : float
        Mean of the first Gaussian component.
    mu2 : float
        Mean of the second Gaussian component.
    sigma : float
        Shared standard deviation of both Gaussian components.
    w : float
        Mixture weight for the first component. Must satisfy 0 < w < 1.
    eps : float, optional
        Lower bound applied to the mixture density to avoid numerical
        underflow when computing responsibilities and ratios.

    Returns
    -------
    np.ndarray of shape (3, N)
        Matrix containing the score evaluated at each sample. The rows
        correspond to the following partial derivatives:

        - Row 0: ∂/∂mu1 log p(x)
        - Row 1: ∂/∂mu2 log p(x)
        - Row 2: ∂/∂w   log p(x)

    Notes
    -----
    • The score with respect to the means is expressed in terms of the
      component responsibilities r1 and r2, reflecting the standard mixture
      model identity:

          ∂/∂mu_k log p(x) = r_k * (x - mu_k) / sigma^2.

    • The score for the mixture weight takes the form:

          ∂/∂w log p(x) = (phi1 - phi2) / p,

      where phi1 and phi2 denote the component densities. This expression
      reflects the sensitivity of the log-likelihood to changes in the mixture
      proportion, and is numerically stable due to the 'eps' safeguard.

    • This score representation is suitable for empirical Fisher computations,
      alignment diagnostics, and spectral analysis of sensitivity operators
      used in information geometry and curvature-based model evaluation.
    """
    x = np.asarray(x)

    # Component densities
    phi1 = _gaussian_pdf(x, mu1, sigma)
    phi2 = _gaussian_pdf(x, mu2, sigma)

    # Mixture density (with stability floor)
    p = w * phi1 + (1.0 - w) * phi2
    p = np.maximum(p, eps)

    # Component responsibilities
    r1 = (w * phi1) / p
    r2 = ((1.0 - w) * phi2) / p

    # Score components for the means
    v_mu1 = r1 * (x - mu1) / (sigma**2)
    v_mu2 = r2 * (x - mu2) / (sigma**2)

    # Score component for the mixture weight
    v_w = (phi1 - phi2) / p

    return np.vstack([v_mu1, v_mu2, v_w])
