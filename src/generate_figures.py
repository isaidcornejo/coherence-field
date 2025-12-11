import os

from src.utils.experiment_io import (
    get_fig_dirs,
    get_results_dir,
    save_results,
    save_spectrum,
)

from src.experiments.gaussian.run_gaussian import run_all_gaussian
from src.experiments.laplace.run_laplace import run_all_laplace
from src.experiments.gmm.run_gmm import run_all_gmm
from src.experiments.mnist.run_mnist import run_mnist_alignment


def main():

    fig_dirs = get_fig_dirs()        # now two directories
    res_dir = get_results_dir()

    print("Guardando figuras en:")
    for d in fig_dirs:
        print("  →", d)

    print("Guardando resultados en:", res_dir)
    for d in fig_dirs:
        os.makedirs(d, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # =========================================================
    # GAUSSIAN
    # =========================================================
    print("\n[1/4] Gaussian experiments...")

    g_eq, g_mis = run_all_gaussian()

    save_spectrum(
        g_eq["lambdas"],
        filename="gaussian_equilibrium.png",
        title="Gaussian – Equilibrium Spectrum"
    )
    save_spectrum(
        g_mis["lambdas"],
        filename="gaussian_misalignment.png",
        title="Gaussian – Misalignment Spectrum"
    )

    save_results(g_eq, "gaussian_equilibrium.npz")
    save_results(g_mis, "gaussian_misalignment.npz")

    # =========================================================
    # LAPLACE
    # =========================================================
    print("\n[2/4] Laplace experiments...")

    l_eq, l_mis = run_all_laplace()

    save_spectrum(
        l_eq["lambdas"],
        filename="laplace_equilibrium.png",
        title="Laplace – Equilibrium Spectrum"
    )
    save_spectrum(
        l_mis["lambdas"],
        filename="laplace_misalignment.png",
        title="Laplace – Misalignment Spectrum"
    )

    save_results(l_eq, "laplace_equilibrium.npz")
    save_results(l_mis, "laplace_misalignment.npz")

    # =========================================================
    # GMM
    # =========================================================
    print("\n[3/4] GMM experiments...")

    gmm_eq, gmm_mis = run_all_gmm()

    save_spectrum(
        gmm_eq["lambdas"],
        filename="gmm_equilibrium.png",
        title="GMM – Equilibrium Spectrum"
    )
    save_spectrum(
        gmm_mis["lambdas"],
        filename="gmm_misalignment.png",
        title="GMM – Misalignment Spectrum"
    )

    save_results(gmm_eq, "gmm_equilibrium.npz")
    save_results(gmm_mis, "gmm_misalignment.npz")

    # =========================================================
    # MNIST
    # =========================================================
    print("\n[4/4] MNIST alignment experiment...")

    mn_out = run_mnist_alignment()

    save_spectrum(
        mn_out["lambdas"],
        filename="mnist_alignment.png",
        title="MNIST – Alignment Spectrum"
    )

    save_results(mn_out, "mnist_alignment.npz")

    print("\n✓ Todas las figuras y resultados generados correctamente.")


if __name__ == "__main__":
    main()
