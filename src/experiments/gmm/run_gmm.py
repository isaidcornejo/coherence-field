from .equilibrium import compute_gmm_equilibrium
from .misalignment import compute_gmm_misalignment


def run_all_gmm():
    """
    Execute the two canonical Gaussian Mixture Model (GMM) experiments:

        1. Equilibrium case:
            Model distribution p matches data distribution q.
            Expected behavior:
                - C ≈ G
                - Alignment operator H ≈ I
                - Scalar alignment A ≈ 0
                - Rectified amplitude φ = 0

        2. Misalignment case:
            Model p differs from data q.
            Expected behavior:
                - C ≠ G
                - Alignment operator H differs from identity
                - Scalar alignment A > 0
                - Rectified amplitude φ > 0

    This function simply orchestrates both computations and returns their
    corresponding diagnostic dictionaries. It performs no numerical work itself.

    Returns:
        tuple(dict, dict):
            - eq:  Output of compute_gmm_equilibrium()
            - mis: Output of compute_gmm_misalignment()

    Notes:
        This is used for:
            - reproducible experiment pipelines
            - figure generation
            - quick sanity checks via CLI
    """
    eq = compute_gmm_equilibrium()
    mis = compute_gmm_misalignment()
    return eq, mis


if __name__ == "__main__":
    # Running as script → perform both experiments and print summary
    eq, mis = run_all_gmm()

    print("GMM equilibrium A =", eq["A"])
    print("GMM misalignment A =", mis["A"])
