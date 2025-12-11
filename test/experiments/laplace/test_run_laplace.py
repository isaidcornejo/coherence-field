from src.experiments.laplace.run_laplace import run_all_laplace


def test_run_all_laplace_returns_two_dicts():
    """
    The run_all_laplace() interface must return two dictionaries:
    one for the equilibrium experiment and one for the misalignment
    experiment. Each dictionary must contain the key diagnostic
    quantities used throughout the alignment pipeline.
    """
    eq, mis = run_all_laplace()

    assert isinstance(eq, dict)
    assert isinstance(mis, dict)

    # Verify presence of minimal diagnostic fields
    for result in (eq, mis):
        for key in ("A", "phi", "G", "C", "H", "lambdas"):
            assert key in result


def test_run_all_laplace_equilibrium_vs_misalignment():
    """
    Correct theoretical behavior of the Laplace distribution under the
    alignment diagnostic:

        • Laplace equilibrium (q = p) does NOT yield A = 0.
          Because Laplace is not a minimal exponential family,
          equilibrium exhibits *structural suppression*:
                λ_i < 1  ⇒  A < 0 and φ = 0.

        • Laplace misalignment with b_data < b_model (default case)
          produces *stronger suppression* than equilibrium:
                A_mis < A_eq < 0

          In this scenario, misalignment does NOT produce reinforcement
          (A > 0). Instead, the Fisher-normalized empirical curvature is
          further reduced, leading to more negative alignment scalars.

    This test checks that the misalignment run intensifies suppression,
    consistent with the analytical predictions and the behavior shown
    in the paper.
    """
    eq, mis = run_all_laplace()

    A_eq = eq["A"]
    A_mis = mis["A"]

    # Equilibrium always yields structural suppression
    assert A_eq < 0

    # Misalignment (with b_data < b_model) increases suppression → A more negative
    assert A_mis < A_eq
