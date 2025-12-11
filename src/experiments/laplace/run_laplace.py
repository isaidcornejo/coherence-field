from .equilibrium import compute_laplace_equilibrium
from .misalignment import compute_laplace_misalignment


def run_all_laplace():
    """
    Execute the two canonical Laplace experiments:

        1. Laplace equilibrium:
            q(x) = p(x | μ, b)
            Expected:
                - C ≠ G   (Laplace is not a minimal exponential family)
                - λ_i < 1 (structural suppression)
                - A < 0
                - φ = 0

        2. Laplace misalignment:
            q(x) ≠ p(x)
            Expected:
                - C departs more strongly from G
                - H deviates further from identity
                - A > 0
                - φ > 0

    This function simply orchestrates both computations and returns them.
    It does not perform mathematical logic by itself, but serves as the
    unified interface for Laplace-related experiments in the pipeline.

    Returns:
        tuple(dict, dict):
            - eq:  output of compute_laplace_equilibrium()
            - mis: output of compute_laplace_misalignment()
    """
    eq = compute_laplace_equilibrium()
    mis = compute_laplace_misalignment()
    return eq, mis


if __name__ == "__main__":
    # When executed directly, print diagnostic summary.
    eq, mis = run_all_laplace()
    print("Laplace equilibrium A =", eq["A"])
    print("Laplace misalignment A =", mis["A"])
