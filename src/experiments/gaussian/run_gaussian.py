from .equilibrium import compute_gaussian_equilibrium
from .misalignment import compute_gaussian_misalignment

def run_all_gaussian():
    eq = compute_gaussian_equilibrium()
    mis = compute_gaussian_misalignment()
    return eq, mis

if __name__ == "__main__":
    eq, mis = run_all_gaussian()
    print("Gaussian equilibrium A =", eq["A"])
    print("Gaussian misalignment A =", mis["A"])
