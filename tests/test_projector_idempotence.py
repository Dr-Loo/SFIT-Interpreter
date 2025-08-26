import numpy as np
from ._helpers import m_projector, m_orthonormal_basis

def test_projector_idempotence_and_symmetry():
    rng = np.random.default_rng(0)
    n, k = 50, 3
    B = m_orthonormal_basis(n, k, M=None, rng=rng)
    P = m_projector(B)  # M=None

    # symmetry & idempotence
    assert np.allclose(P, P.T, atol=1e-12)
    assert np.allclose(P @ P, P, atol=1e-12)

    # Px = x for x in span(B)
    coeffs = rng.normal(size=(k,))
    x_in = B @ coeffs
    assert np.allclose(P @ x_in, x_in, atol=1e-12)

    # P annihilates orthogonal complement
    # Build full orthonormal basis whose first k columns are B
    # (this ensures Bperp spans the orthogonal complement of span(B))
    Q_full, _ = np.linalg.qr(
        np.concatenate([B, rng.standard_normal((n, n - k))], axis=1)
    )
    Bperp = Q_full[:, k:]
    y = Bperp @ rng.normal(size=(n - k,))
    assert np.linalg.norm(P @ y) < 1e-10
