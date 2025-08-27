import numpy as np

def spectral_projector_kernel(A, tol=1e-10):
    # Projector onto (numerical) kernel of A via eigendecomposition
    w, V = np.linalg.eigh(A)
    mask = w < tol
    if not np.any(mask):
        # no kernel; return zero projector
        n = A.shape[0]
        return np.zeros((n, n))
    B = V[:, mask]                  # basis for kernel
    return B @ B.T                  # M = I case

def test_davis_kahan_budget_kernel_block():
    rng = np.random.default_rng(1)
    n, k = 40, 3

    # Build an operator A with kernel of dimension k and gap ~ 1
    # Let Q = [B | C] orthonormal; set A = C C^T so:
    #  - eigenvalues: 0 (mult=k), 1 (mult=n-k)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    B = Q[:, :k]
    C = Q[:, k:]
    A = C @ C.T
    gap = 1.0  # first positive eigenvalue

    # Perturbation E with small spectral norm
    G = rng.normal(size=(n, n))
    E = 1e-3 * 0.5 * (G + G.T)       # symmetric small perturbation
    A_tilde = A + E

    # Projectors onto kernels before/after
    P = spectral_projector_kernel(A)
    P_tilde = spectral_projector_kernel(A_tilde)

    # Operator norms (2-norm)
    E_norm = np.linalg.norm(E, 2)
    drift = np.linalg.norm(P - P_tilde, 2)

    # Davis–Kahan-style bound: drift ≤ ||E|| / gap (up to small numerical slack)
    assert drift <= E_norm / gap + 5e-3
