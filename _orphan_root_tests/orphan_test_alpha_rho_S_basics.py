import numpy as np
from _helpers import make_M, m_orthonormal_basis, shannon_entropy

def test_alpha_rho_S_basics():
    dim, k = 64, 3
    M = make_M(dim)
    B = m_orthonormal_basis(dim, k, M)  # M-orthonormal columns

    expected_alpha = np.array([0.6, 0.3, 0.1], dtype=float)
    # construct v = sum_i alpha_i * B[:, i]
    v = B @ expected_alpha

    # alpha = B^T M v (because B is M-orthonormal)
    alpha = B.T @ M @ v
    rho = (alpha**2) / np.sum(alpha**2)
    S = shannon_entropy(rho)

    expected_rho = expected_alpha**2 / np.sum(expected_alpha**2)
    expected_S = shannon_entropy(expected_rho)

    assert np.allclose(alpha, expected_alpha, atol=1e-10)
    assert np.allclose(rho, expected_rho, atol=1e-12)
    assert abs(S - expected_S) < 1e-12
