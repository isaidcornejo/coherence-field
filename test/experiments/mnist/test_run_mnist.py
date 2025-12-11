from src.experiments.mnist.run_mnist import run_mnist, main


def test_run_mnist_function_exists():
    """
    The MNIST experiment module must expose a callable wrapper function
    `run_mnist()`, which orchestrates the end-to-end workflow for:

        • loading data,
        • constructing the model,
        • computing score vectors,
        • estimating alignment diagnostics,
        • and optionally logging or saving results.

    This smoke test ensures that the function is correctly defined and importable
    as part of the experiment interface.
    """
    assert callable(run_mnist)


def test_run_mnist_main_exists():
    """
    The module must also define a `main()` entrypoint, enabling the MNIST
    experiment pipeline to be executed as a standalone script via:

        python -m src.experiments.mnist.run_mnist

    or through direct CLI invocation.

    This test confirms that the entrypoint is present and callable, ensuring
    proper integration with external automation, experiment runners, and
    reproducibility frameworks.
    """
    assert callable(main)
