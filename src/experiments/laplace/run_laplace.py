from .equilibrium import compute_laplace_equilibrium
from .misalignment import compute_laplace_misalignment

def run_all_laplace():
    eq = compute_laplace_equilibrium()
    mis = compute_laplace_misalignment()
    return eq, mis

if __name__ == "__main__":
    eq, mis = run_all_laplace()
    print("Laplace equilibrium A =", eq["A"])
    print("Laplace misalignment A =", mis["A"])
