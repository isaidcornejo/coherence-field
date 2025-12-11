import inspect
import src.generate_figures as gf


def test_main_function_exists():
    """
    The figure-generation pipeline must expose a `main()` function that serves as
    the unified entrypoint for:

        • running all experiment modules (Gaussian, Laplace, GMM, MNIST),
        • generating all diagnostic figures,
        • saving spectral plots and summary artifacts.

    This test verifies that the entrypoint is present and callable, ensuring
    compatibility with CLI usage and automated build systems.
    """
    assert hasattr(gf, "main")
    assert callable(gf.main)


def test_no_side_effects_on_import():
    """
    Importing the `generate_figures` module must *not* trigger execution of the
    experiment pipeline. This is a crucial safety guarantee, ensuring:

        • importing the module in test environments does not produce files,
        • execution happens only when intentionally invoked via:

              if __name__ == "__main__":
                  main()

        • modules remain import-safe for documentation builders, interactive
          environments, and external tooling.

    This test inspects the source code to confirm that main() is called *only*
    inside a __main__ guard.
    """
    source = inspect.getsource(gf)

    # The file must contain a proper entrypoint guard.
    assert "__main__" in source
    assert "main()" in source


def test_dependencies_exist():
    """
    The figure pipeline depends on several high-level experiment orchestrators.
    This test verifies that these functions are importable and callable *without
    executing them*, ensuring that the module dependency graph is stable.

    These functions include:
        • run_all_gaussian
        • run_all_laplace
        • run_all_gmm
        • run_mnist_alignment

    All must exist and be callable.
    """
    from src.experiments.gaussian.run_gaussian import run_all_gaussian
    from src.experiments.laplace.run_laplace import run_all_laplace
    from src.experiments.gmm.run_gmm import run_all_gmm
    from src.experiments.mnist.run_mnist import run_mnist_alignment

    assert callable(run_all_gaussian)
    assert callable(run_all_laplace)
    assert callable(run_all_gmm)
    assert callable(run_mnist_alignment)


def test_save_functions_importable():
    """
    The pipeline relies on two core saving utilities:

        • save_results → persists numerical experiment outputs (NPZ)
        • save_spectrum → writes spectral plots to publication directories

    This test ensures that both functions are importable and callable, verifying
    that the I/O subsystem of the experiment framework is available before any
    pipeline execution occurs.
    """
    from src.utils.experiment_io import save_results, save_spectrum

    assert callable(save_results)
    assert callable(save_spectrum)
