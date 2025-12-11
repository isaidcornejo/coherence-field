from .alignment import run_mnist_alignment


def main():
    """
    Entry point for running the full MNIST alignment experiment.

    This function:
        • Executes the Fisher–Empirical MNIST pipeline.
        • Prints the scalar diagnostics A and φ.
        • Returns the full experiment dictionary for downstream analysis.

    Intended use:
        $ python run_mnist.py
    """
    out = run_mnist_alignment()
    print("MNIST alignment A =", out["A"])
    print("MNIST alignment phi =", out["phi"])
    return out


def run_mnist():
    """
    Convenience wrapper so that external code can call:

        results = run_mnist()

    without needing to import run_mnist_alignment directly.
    """
    return run_mnist_alignment()


if __name__ == "__main__":
    # When executed as a script, run the experiment and print a summary.
    main()
