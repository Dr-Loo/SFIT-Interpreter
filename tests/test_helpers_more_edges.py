import numpy as np
import pytest

from ._helpers import make_M, normalize_weights, shannon_entropy, m_orthonormal_basis

def test_make_M_with_weight_vector_path():
    n = 8
    w = 10.0 ** np.linspace(-2, 2, n)  # positive, non-uniform
    M = make_M(n, w)
    assert M.shape == (n, n)
    assert np.allclose(np.diag(M), w)
    x = np.arange(1, n + 1, dtype=float)
    assert x @ (M @ x) > 0.0  # SPD check

def test_normalize_weights_raises_on_nonpositive_sum():
    w = np.zeros(5)  # sum == 0 -> guard
    with pytest.raises((ValueError, ZeroDivisionError)):
        _ = normalize_weights(w)

def test_normalize_weights_raises_on_negative_entries():
    w = np.array([0.2, -0.1, 0.9])
    with pytest.raises(ValueError):
        _ = normalize_weights(w)

def test_shannon_entropy_rejects_bad_shape():
    rho_bad = np.array([[0.5, 0.5]])  # 2D -> guard
    with pytest.raises((ValueError, TypeError)):
        _ = shannon_entropy(rho_bad)

def test_m_orthonormal_basis_with_identity_metric():
    rng = np.random.default_rng(0)
    n, k = 20, 3
    A = rng.normal(size=(n, k))
    M = make_M(n)  # identity
    B = m_orthonormal_basis(A, M)
    assert B.shape[0] == n and 1 <= B.shape[1] <= k
    G = B.T @ (M @ B)
    assert np.allclose(G, np.eye(B.shape[1]), atol=1e-8)
