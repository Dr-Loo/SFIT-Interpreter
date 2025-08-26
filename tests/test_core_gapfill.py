import numpy as np
from numpy.linalg import norm
import pytest

# hit the public API from the installed package
from sfit_sgf.core import (
    make_M,
    m_orthonormal_basis,
    m_projector,
    projector,                 # alias coverage
    shannon_entropy,
    davis_kahan_synthetic,
)

def test_make_M_accepts_full_spd_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    W = A.T @ A + 1e-3 * np.eye(5)   # SPD
    M = make_M(5, W)                 # 2-D weights branch
    assert np.allclose(M, W)

def test_m_orthonormal_basis_positional_M_and_no_prefilter():
    # positional-M branch: m_orthonormal_basis(A, M)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 3))
    w = 10.0 ** rng.uniform(-2, 2, size=20)
    M = make_M(20, w)

    # positional M triggers the "k is actually M" logic
    B = m_orthonormal_basis(A, M)
    G = B.T @ (M @ B)
    assert np.allclose(G, np.eye(B.shape[1]), atol=1e-10)

    # also exercise prefilter=False path explicitly
    B2 = m_orthonormal_basis(A, k=3, M=M, prefilter=False)
    assert np.allclose(B2.T @ (M @ B2), np.eye(B2.shape[1]), atol=1e-10)

def test_cholesky_fallback_path():
    # PSD (rank-1) -> Cholesky fails -> eigen fallback in _chol_from_spd
    M = np.array([[1.0, 1.0],
                  [1.0, 1.0]])
    A = np.eye(2)  # any 2×2 basis candidate
    B = m_orthonormal_basis(A, M)
    assert B.shape[0]==2 and 1 <= B.shape[1] <= 2
    G = B.T @ (M @ B)
    assert np.allclose(G, np.eye(B.shape[1]), atol=1e-10)

def test_projector_alias_and_weighted_idempotence():
    rng = np.random.default_rng(3)
    n, k = 30, 4
    w = 10.0 ** rng.uniform(-2, 2, size=n)
    M = make_M(n, w)
    A = rng.normal(size=(n, k))
    B = m_orthonormal_basis(A, M)
    P = projector(B, M)            # alias coverage
    # weighted idempotence: P^2 = P
    assert np.allclose(P @ P, P, atol=1e-10)

def test_dk_return_full_branch_and_keys():
    out = davis_kahan_synthetic(n=24, k=3, eps=1e-4, return_full=True)
    for key in ("A", "A_prime", "P", "P_prime", "E", "gap", "E_norm", "drift"):
        assert key in out
