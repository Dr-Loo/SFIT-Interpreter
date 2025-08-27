import numpy as np
from ._helpers import m_orthonormal_basis, normalize_weights, shannon_entropy

def test_alpha_rho_S_basics():
    rng = np.random.default_rng(0)
    n, k = 192, 3
    B = m_orthonormal_basis(n, k, M=None, rng=rng)

    alpha_true = np.array([0.6, 0.3, 0.1], dtype=float)
    v = B @ alpha_true

    alpha = B.T @ v
    rho = normalize_weights(alpha**2)
    S = shannon_entropy(rho)

    rho_true = alpha_true**2 / (alpha_true**2).sum()
    S_true = -np.sum(rho_true * np.log(rho_true))

    assert np.allclose(alpha, alpha_true, atol=1e-12)
    assert np.allclose(rho, rho_true, atol=1e-12)
    assert abs(S - S_true) < 1e-12
