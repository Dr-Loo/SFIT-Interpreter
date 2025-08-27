import numpy as np
import pytest
from sfit_sgf.core import (
    make_M, normalize_weights, shannon_entropy,
    m_orthonormal_basis, m_projector, davis_kahan_synthetic
)

def test_make_M_bad_diag_and_shape():
    # negative / zero diagonals should raise
    with pytest.raises(ValueError):
        make_M(3, weights=np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        make_M(3, weights=np.array([1.0, -1.0, 2.0]))
    # wrong 2D shape should raise
    with pytest.raises(ValueError):
        make_M(3, weights=np.ones((2, 3)))

def test_m_orthonormal_basis_generator_requires_k():
    # hits the "generator form requires k" branch
    with pytest.raises(TypeError):
        m_orthonormal_basis(5)

def test_m_orthonormal_basis_drop_tiny_columns():
    # forces the "keep none" path -> returns n x 0
    n = 5
    A = np.zeros((n, 2))
    B = m_orthonormal_basis(A, k=2, M=None, tol=1.1)  # tol > any norm
    assert B.shape == (n, 0)

def test_m_orthonormal_basis_positional_M_signature():
    # exercise the branch where M is passed positionally as the 2nd arg
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 3))
    M = make_M(8)
    B1 = m_orthonormal_basis(A, M)        # M in the k-position
    B2 = m_orthonormal_basis(A, M=M)      # explicit M kwarg
    G1 = B1.T @ (M @ B1)
    G2 = B2.T @ (M @ B2)
    assert np.allclose(G1, np.eye(B1.shape[1]), atol=1e-10)
    assert np.allclose(G2, np.eye(B2.shape[1]), atol=1e-10)

def test_projector_identity_branch():
    # M=None branch of projector
    rng = np.random.default_rng(0)
    A = rng.normal(size=(10, 3))
    B = m_orthonormal_basis(A, M=None)
    P = m_projector(B, M=None)
    assert np.allclose(P @ P, P, atol=1e-12)

def test_projector_weighted_branch():
    # weighted projector branch (SPD M)
    rng = np.random.default_rng(2)
    A = rng.normal(size=(12, 4))
    w = 10.0 ** rng.uniform(-2, 2, size=12)
    M = make_M(12, w)
    B = m_orthonormal_basis(A, M=M)
    P = m_projector(B, M)
    x = rng.normal(size=12)
    assert np.allclose(P @ (P @ x), P @ x, atol=1e-10)

def test_davis_kahan_return_full_and_eps_alias():
    # hits eps alias + return_full True branch
    out = davis_kahan_synthetic(n=20, k=2, gap=1.0, eps=5e-4, return_full=True)
    assert isinstance(out, dict)
    assert "P" in out and "P_prime" in out
    assert pytest.approx(out["E_norm"], rel=0, abs=1e-15) == 5e-4

def test_entropy_and_normalize_guards():
    # extra guard branches for completeness
    with pytest.raises(ValueError):
        normalize_weights(np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        normalize_weights(np.array([1, -0.1, 0.1]))
    with pytest.raises(ValueError):
        shannon_entropy(np.array([[0.5, 0.5]]))  # bad shape
