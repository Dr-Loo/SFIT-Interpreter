import numpy as np
from sfit_sgf.core import m_orthonormal_basis, normalize_weights, shannon_entropy

def main():
    rng = np.random.default_rng(0)
    n, k = 192, 3

    # random M-orthonormal basis (M = I by default)
    B = m_orthonormal_basis(n, k, rng=rng)

    # ground-truth memory coefficients
    alpha_true = np.array([0.6, 0.3, 0.1], dtype=float)

    # signal with pure harmonic content
    v = B @ alpha_true

    # recover alpha, compute rho and entropy
    alpha = B.T @ v
    rho = normalize_weights(alpha**2)
    S = shannon_entropy(rho)

    print("alpha:", np.round(alpha, 10))
    print("rho:", np.round(rho, 8))
    print("S:", S)

    rho_true = alpha_true**2 / np.sum(alpha_true**2)
    print("rho_expected:", np.round(rho_true, 8))

if __name__ == "__main__":
    main()
