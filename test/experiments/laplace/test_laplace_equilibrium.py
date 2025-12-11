import numpy as np
from numpy.testing import assert_allclose

from src.experiments.laplace.equilibrium import compute_laplace_equilibrium


def test_laplace_equilibrium_keys_present():
    """
    The Laplace equilibrium routine must return all core diagnostic fields
    used by the alignment pipeline. These fields include:
        - G       : analytic Fisher information
        - C       : empirical score covariance under q = p
        - H       : Fisher-normalized alignment operator
        - lambdas : eigenvalues of H
        - A       : scalar alignment deviation
        - phi     : rectified amplitude
        - mu, b   : model parameters
        - num_samples : Monte Carlo sample count
    """
    result = compute_laplace_equilibrium(num_samples=40_000)

    expected = {
        "G", "C", "H", "lambdas", "A", "phi",
        "mu", "b", "num_samples"
    }
    assert expected.issubset(result.keys())


def test_laplace_equilibrium_G_shape_and_values():
    """
    The Fisher information matrix for the univariate Laplace distribution
    has a closed-form analytic structure:

        I(mu) = 1 / b^2
        I(b)  = 2 / b^2

    with zero off-diagonal terms. This test verifies that the implementation
    of compute_laplace_equilibrium() respects this theoretical identity.
    """
    b = 2.0
    result = compute_laplace_equilibrium(num_samples=20_000, b=b)

    G = result["G"]
    expected_G = np.array([
        [1 / b**2, 0],
        [0,        2 / b**2]
    ])

    assert_allclose(G, expected_G, atol=1e-12)


def test_laplace_equilibrium_C_differs_from_G():
    """
    Key property of the Laplace distribution:

        Even under equilibrium (q = p), the empirical score covariance C
        does *not* equal the Fisher information G.

    This differs fundamentally from minimal exponential families (e.g. Gaussian),
    where C = G exactly when q = p.

    The mismatch C ≠ G is a structural property of the Laplace model and is
    responsible for its intrinsic suppression behavior in the alignment spectrum.
    """
    result = compute_laplace_equilibrium(num_samples=200_000)

    G = result["G"]
    C = result["C"]

    # They should not be close; this verifies the structural mismatch.
    assert not np.allclose(C, G, rtol=0.05, atol=0.05)


def test_laplace_equilibrium_A_is_negative():
    """
    Laplace equilibrium occupies a *structural suppression* regime.

    Because C < G in the Fisher-normalized sense, the eigenvalues of the
    alignment operator H satisfy:

        λ_i < 1   for all i

    leading to:

        A = Σ_i (λ_i − 1) < 0

    This is a central theoretical signature of the Laplace family.
    """
    result = compute_laplace_equilibrium(num_samples=200_000)
    A = result["A"]

    assert A < 0


def test_laplace_equilibrium_phi_zero():
    """
    For A < 0, the rectified amplitude is defined as:

        φ = max{ sqrt(A), 0 } = 0

    Thus φ must vanish identically in the suppression regime. This provides a
    clean binary separation between reinforcement (A > 0, φ > 0) and suppression
    (A < 0, φ = 0).
    """
    result = compute_laplace_equilibrium(num_samples=200_000)
    assert result["phi"] == 0.0


def test_laplace_equilibrium_H_symmetric():
    """
    Although H is implemented computationally as H = G^{-1} C, it is guaranteed
    to be similar to the symmetric matrix G^{-1/2} C G^{-1/2}. Therefore, H must
    be symmetric up to numerical precision.

    This test ensures the eigenstructure is well-defined and the implementation
    is consistent with the underlying information-geometric theory.
    """
    result = compute_laplace_equilibrium(num_samples=200_000)
    H = result["H"]

    assert_allclose(H, H.T, atol=1e-6)
