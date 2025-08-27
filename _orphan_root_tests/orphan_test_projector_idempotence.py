import numpy as np

def m_projector(B, M=None):
    """
    M-orthogonal projector onto span(B).
    B: (n,k) with M-orthonormal columns if M=I.
    """
    n, k = B.shape
    if M is None:
        M = np.eye(n)
    G = B.T @ M @ B                     # k x k Gram
    return B @ np.linalg.inv(G) @ (B.T @ M)

def test_projector_idempotence_and_symmetry():
    rng = np.random.default_rng(0)
    n, k = 50, 3
    X = rng.normal(size=(n, k))
    # M = I; make B orthonormal via QR
    B, _ = np.linalg.qr(X)
    P = m_projector(B)

    # Symmetry (under standard inner product since M=I)
    assert np.allclose(P, P.T, atol=1e-10)

    # Idempotence
    assert np.allclose(P @ P, P, atol=1e-10)

    # Projects correctly: Px = x for x in span(B)
    coeffs = rng.normal(size=(k,))
    x_in = B @ coeffs
    assert np.allclose(P @ x_in, x_in, atol=1e-10)

    # Orthogonality: P annihilates orthogonal complement
    # Build y in orthogonal complement via QR complete
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Bperp = Q[:, k:]
    y = Bperp @ rng.normal(size=(n-k,))
    assert np.linalg.norm(P @ y) < 1e-10
