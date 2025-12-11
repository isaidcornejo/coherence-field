from src.experiments.gmm.run_gmm import run_all_gmm


def test_run_all_gmm_returns_two_dicts():
    """
    The high-level GMM runner must execute the two canonical experiments:

        (1) Gaussian Mixture Equilibrium (q = p)
        (2) Gaussian Mixture Misalignment (q ≠ p)

    and return a tuple (eq, mis), where each entry is a dictionary containing
    the core alignment diagnostics:

        - A        : scalar alignment deviation
        - phi      : rectified amplitude
        - G, C     : Fisher matrix and empirical score covariance
        - H        : alignment operator H = G^{-1} C
        - lambdas  : eigenvalues of H

    This structural guarantee ensures that downstream experiments, spectral
    analysis, and visualization tools can treat run_all_gmm() as a stable API.
    """
    eq, mis = run_all_gmm()

    assert isinstance(eq, dict)
    assert isinstance(mis, dict)

    # Verify presence of all critical diagnostic fields
    for result in (eq, mis):
        for key in ("A", "phi", "G", "C", "H", "lambdas"):
            assert key in result


def test_run_all_gmm_equilibrium_vs_misalignment():
    """
    Qualitative expected behavior for Gaussian Mixture Models:

        • Equilibrium (q = p)
            - Score covariance matches Fisher curvature
            - Alignment operator eigenvalues satisfy λ_i ≈ 1
            - Therefore A = Σ_i (λ_i − 1) ≈ 0
            - phi is small or zero, depending on small positive noise in A

        • Misalignment (q ≠ p)
            - Data-induced curvature departs from Fisher geometry
            - At least one eigenvalue λ_i > 1 (reinforcement)
            - Therefore A becomes strictly positive
            - phi > 0 by definition (φ = max{sqrt(A), 0})

    This test enforces the expected *ordering* of alignment magnitudes:

            |A_eq|  <  small threshold
             A_mis >  A_eq

    rather than requiring exact numerical values, which would depend on
    Monte Carlo variance, sampling scale, and parameter configuration.
    """
    eq, mis = run_all_gmm()

    A_eq = eq["A"]
    A_mis = mis["A"]

    # Equilibrium must produce a scalar deviation close to zero.
    assert abs(A_eq) < 0.1

    # Misalignment must reflect stronger curvature deformation.
    assert A_mis > A_eq
