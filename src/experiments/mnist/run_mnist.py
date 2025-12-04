from .alignment import run_mnist_alignment


def main():
    out = run_mnist_alignment()
    print("MNIST alignment A =", out["A"])
    print("MNIST alignment phi =", out["phi"])
    return out


def run_mnist():
    return run_mnist_alignment()

if __name__ == "__main__":
    main()
