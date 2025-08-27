import numpy as np
from sfit_sgf.core import make_M, m_orthonormal_basis, m_projector

def main():
    rng = np.random.default_rng(3)
    n, k = 40, 4
    # strongly non-uniform diagonal SPD metric
    w = 10.0 ** rng.uniform(-2, 2, size=n)
    M = make_M(n, w)

    A = rng.normal(size=(n, k))
    B = m_orthonormal_basis(A, M=M)
    P = m_projector(B, M)

    # checks (for general M, P is NOT symmetric in the Euclidean sense)
    print("Idempotent (P@P ≈ P):", np.allclose(P @ P, P, atol=1e-10))
    print("M-self-adjoint (P^T M ≈ M P):", np.allclose(P.T @ M, M @ P, atol=1e-10))

    # project a random vector and check it lands in span(B) in the M-inner product
    x = rng.normal(size=n)
    Px = P @ x
    res = (np.eye(n) - P) @ Px
    print("|| (I-P)Px || (tiny):", np.linalg.norm(res))

if __name__ == "__main__":
    main()
