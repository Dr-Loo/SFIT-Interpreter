import numpy as np
from numpy.linalg import norm

# ----------------------------
# Metrics / normalization
# ----------------------------

def make_M(n: int, weights=None):
    """
    Build an SPD metric matrix M (n x n).
    - If weights is None: identity
    - If weights is 1D: diagonal with those entries (must be > 0)
    - If weights is 2D (n x n): returned as-is (assumed SPD)
    """
    if weights is None:
        return np.eye(n, dtype=float)
    w = np.asarray(weights)
    if w.ndim == 1:
        if (w <= 0).any():
            raise ValueError("All diagonal weights must be positive for SPD.")
        return np.diag(w.astype(float))
    if w.shape != (n, n):
        raise ValueError("weights must be length-n or n×n.")
    return w.astype(float)


def normalize_weights(w):
    """
    Normalize a 1D, nonnegative array to sum 1.
    Raises on bad shape, negatives, or nonpositive sum.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("normalize_weights expects a 1D array.")
    if (w < 0).any():
        raise ValueError("Weights must be nonnegative.")
    s = w.sum()
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    return w / s


def shannon_entropy(rho):
    """
    Shannon entropy H(rho) = -sum rho_i log rho_i.
    Accepts unnormalized nonnegative weights and normalizes internally.
    Rejects bad shapes and negatives.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1:
        raise ValueError("shannon_entropy expects a 1D array.")
    if (rho < 0).any():
        raise ValueError("Probabilities/weights must be nonnegative.")
    if rho.sum() <= 0:
        raise ValueError("Total weight must be positive.")
    p = rho / rho.sum()
    p = np.clip(p, 0.0, 1.0)  # numerical safety
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


# ----------------------------
# Orthonormal bases / projectors
# ----------------------------

def _chol_from_spd(M):
    """Robust factor L with M = L L^T for SPD / nearly-SPD M."""
    M = 0.5 * (M + M.T)
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(M)
        w = np.maximum(w, 1e-15)
        return U @ np.diag(np.sqrt(w))


def m_orthonormal_basis(
    A_or_n,
    k=None,
    M=None,
    tol: float = 1e-12,
    reorth: bool = True,
    rng=None,
    prefilter: bool = True,
):
    """
    Flexible M-orthonormalizer / generator.

    Forms supported:
      1) B = m_orthonormal_basis(A, M)             # orthonormalize columns of A
      2) B = m_orthonormal_basis(A, k, M)          # optionally truncate to k
      3) B = m_orthonormal_basis(n, k, M=None, rng=rng)  # generate random basis (n×k)

    Returns B with B^T M B = I_r (r <= min(n,k,rank)).
    """
    # Case 3: generator form (n, k, ...):
    if isinstance(A_or_n, (int, np.integer)):
        n = int(A_or_n)
        if k is None:
            raise TypeError("Generator form requires k.")
        k = int(k)
        if rng is None:
            rng = np.random.default_rng()
        A = rng.normal(size=(n, k))
    else:
        # Case 1/2: array-first API
        A = np.asarray(A_or_n, dtype=float)
        # If caller used positional M: m_orthonormal_basis(A, M)
        if k is not None and np.ndim(k) == 2 and k.shape == (A.shape[0], A.shape[0]):
            M, k = k, None

    n = A.shape[0]
    if M is None:
        M = np.eye(n, dtype=float)
    else:
        M = np.asarray(M, dtype=float)

    # Decide target columns
    if k is None:
        k = A.shape[1]
    k = int(min(k, A.shape[1], n))
    A = A[:, :k].copy()

    # Optional prefilter: drop columns with tiny M-norm before QR
    if prefilter and A.size:
        keep0 = []
        for j in range(A.shape[1]):
            aj = A[:, j]
            if aj.T @ (M @ aj) >= tol:
                keep0.append(j)
        if keep0:
            A = A[:, keep0]
        else:
            return np.zeros((n, 0), dtype=float)

    # Orthonormalize in M using Cholesky trick: M = L L^T
    L = _chol_from_spd(M)
    Btilde = L.T @ A
    Q, _ = np.linalg.qr(Btilde, mode="reduced")
    if reorth:
        Q, _ = np.linalg.qr(Q, mode="reduced")
    # Map back: solve L^T B = Q
    B = np.linalg.solve(L.T, Q)

    # Drop columns with tiny M-norm (post-filter)
    keep = [j for j in range(B.shape[1]) if B[:, j].T @ (M @ B[:, j]) >= tol]
    if not keep:
        return np.zeros((n, 0), dtype=float)
    B = B[:, keep]

    # Final polish so that B^T M B ≈ I
    G = B.T @ (M @ B)
    w, U = np.linalg.eigh(0.5 * (G + G.T))
    w = np.maximum(w, 1e-15)
    Gm12 = U @ np.diag(w**-0.5) @ U.T
    return B @ Gm12


def m_projector(B, M=None):
    """
    Return the M-orthogonal projector onto span(B).

    If M is None (Euclidean), P = Q Q^T with Q = qr(B).
    For SPD M: let M = L L^T, Bh = L^T B, Qh = qr(Bh), Q = L^{-T} Qh, then P = Q Q^T M.
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    if B.size == 0:
        return np.zeros((n, n), dtype=float)

    if M is None:
        Q, _ = np.linalg.qr(B, mode="reduced")
        P = Q @ Q.T
        return 0.5 * (P + P.T)  # enforce symmetry numerically

    M = np.asarray(M, dtype=float)
    L = _chol_from_spd(M)
    Bh = L.T @ B
    Qh, _ = np.linalg.qr(Bh, mode="reduced")
    Q = np.linalg.solve(L.T, Qh)
    P = Q @ (Q.T @ M)  # M-self-adjoint projector
    return P


# ----------------------------
# Davis–Kahan helper
# ----------------------------

def davis_kahan_synthetic(
    n=50, k=3, gap=1.0, E_norm=None, eps=None, seed=0, M=None, return_full=False
):
    """
    Construct a simple symmetric pencil for Davis–Kahan experiments.

    Returns either:
      (drift, E_norm, gap)               [default]
    or a dict if return_full=True.

    Accepts `eps` as alias for `E_norm`.
    """
    if E_norm is None:
        E_norm = eps if eps is not None else 1e-3

    rng = np.random.default_rng(seed)
    A = np.diag(np.concatenate([np.zeros(k), np.full(n - k, float(gap))]))

    R = rng.normal(size=(n, n))
    E = 0.5 * (R + R.T)
    s = norm(E, 2)
    E *= (E_norm / (s if s != 0 else 1.0))
    A_prime = A + E

    # True unperturbed subspace and projector
    B = np.eye(n, k)
    P = m_projector(B, M)

    # Perturbed: take k smallest eigenvectors
    w, V = np.linalg.eigh(A_prime)
    idx = np.argsort(w)[:k]
    Bp = V[:, idx]
    Pp = m_projector(Bp, M)

    drift = norm(P - Pp, 2)

    if return_full:
        return {
            "A": A,
            "A_prime": A_prime,
            "P": P,
            "P_prime": Pp,
            "E": E,
            "gap": float(gap),
            "E_norm": float(E_norm),
            "drift": float(drift),
        }
    return float(drift), float(E_norm), float(gap)


__all__ = [
    "make_M",
    "normalize_weights",
    "shannon_entropy",
    "m_orthonormal_basis",
    "m_projector",
    "projector",              # <— add this
    "davis_kahan_synthetic",
]

# Back-compat alias so tests can `from sfit_sgf.core import projector`
projector = m_projector
