from .equilibrium import compute_gaussian_equilibrium
from .misalignment import compute_gaussian_misalignment


def run_all_gaussian():
    """
    Run the two canonical Gaussian experiments:

        1. Fisher-equilibrium case (q = p)
        2. Misalignment case (q ≠ p)

    This function simply orchestrates the two computations and returns
    their full result dictionaries.

    Returns:
        tuple(dict, dict):
            - eq:  Output dictionary from compute_gaussian_equilibrium()
            - mis: Output dictionary from compute_gaussian_misalignment()

    Notes:
        - This function is designed for reproducibility pipelines and for
          batch figure generation in the Gaussian section of the paper.
        - It intentionally performs no analysis itself; downstream code
          inspects fields like A, φ, eigenvalues, etc.
    """
    eq = compute_gaussian_equilibrium()
    mis = compute_gaussian_misalignment()
    return eq, mis


if __name__ == "__main__":
    # If run as a script, execute both experiments and print diagnostics.
    eq, mis = run_all_gaussian()

    print("Gaussian equilibrium A =", eq["A"])
    print("Gaussian misalignment A =", mis["A"])
