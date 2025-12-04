from .equilibrium import compute_gmm_equilibrium
from .misalignment import compute_gmm_misalignment

def run_all_gmm():
    eq = compute_gmm_equilibrium()
    mis = compute_gmm_misalignment()
    return eq, mis

if __name__ == "__main__":
    eq, mis = run_all_gmm()
    print("GMM equilibrium A =", eq["A"])
    print("GMM misalignment A =", mis["A"])
