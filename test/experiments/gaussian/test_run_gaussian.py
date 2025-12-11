from src.experiments.gaussian.run_gaussian import run_all_gaussian


def test_run_all_gaussian_basic():
    """
    The run_all_gaussian() interface is a high-level orchestration function that
    executes two canonical Gaussian experiments:

        1. Gaussian equilibrium (q = p)
        2. Gaussian misalignment (q ≠ p)

    It must return two dictionaries—one for each experiment—each containing the
    core alignment diagnostics:

        - A        : scalar alignment deviation
        - phi      : rectified amplitude
        - G, C     : Fisher matrix and empirical score covariance
        - H        : alignment operator G^{-1}C
        - lambdas  : eigenvalues of H

    This test verifies correct structural output and ensures downstream
    diagnostics can rely on run_all_gaussian() as a stable API boundary.
    """
    eq, mis = run_all_gaussian()

    assert isinstance(eq, dict)
    assert isinstance(mis, dict)

    for result in (eq, mis):
        for key in ("A", "phi", "G", "C", "H", "lambdas"):
            assert key in result


def test_run_all_gaussian_equilibrium_vs_misalignment():
    """
    Qualitative alignment behavior for Gaussian models:

        • Equilibrium (q = p)
              → empirical curvature matches Fisher curvature
              → eigenvalues λ_i ≈ 1
              → A = Σ_i (λ_i − 1) ≈ 0
              → φ is small (sampling noise scale)

        • Misalignment (q ≠ p)
              → curvature becomes anisotropic
              → at least one eigenvalue satisfies λ_i > 1
              → A becomes strictly positive
              → φ > 0

    Because Monte Carlo fluctuations vary from run to run, this test enforces
    qualitative—not exact—comparisons:

        |A_eq| < small threshold      (near zero)
        A_mis  > A_eq                 (reflecting reinforcement under mismatch)

    This ensures that run_all_gaussian() produces outputs consistent with the
    theoretical alignment structure of the Gaussian family.
    """
    eq, mis = run_all_gaussian()

    # Equilibrium must be close to zero, within reasonable sampling noise.
    assert abs(eq["A"]) < 0.2

    # Misalignment must produce a larger scalar deviation than equilibrium.
    assert mis["A"] > eq["A"]
