# Fixed the Python syntax (replaced '&&' with 'and')
import numpy as np

EPS = 1e-16

def shannon_entropy(weights):
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative")
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    return float(-np.sum(p * np.log(p + EPS)))

def m_inner(x, y, M=None):
    if M is None:
        return float(np.dot(x, y))
    return float(x @ M @ y)

def m_norm(x, M=None):
    return np.sqrt(max(m_inner(x, x, M), 0.0))

def m_projector(B, M=None):
    B = np.asarray(B, dtype=float)
    if M is None:
        G = B.T @ B
        Ginv = np.linalg.pinv(G)
        return B @ Ginv @ B.T
    else:
        MB = M @ B
        G = B.T @ MB
        Ginv = np.linalg.pinv(G)
        return B @ Ginv @ (B.T @ M)

def m_orthonormal_basis(X, M=None):
    X = np.array(X, dtype=float, copy=True)
    n, k = X.shape
    Q = np.zeros_like(X)
    for i in range(k):
        v = X[:, i].copy()
        for j in range(i):
            proj = m_inner(Q[:, j], v, M)
            v -= proj * Q[:, j]
        nv = m_norm(v, M)
        if nv < 1e-14:
            r = np.random.default_rng(i).normal(size=n)
            for j in range(i):
                r -= m_inner(Q[:, j], r, M) * Q[:, j]
            nv = m_norm(r, M)
            v = r / (nv + EPS)
        else:
            v /= nv
        Q[:, i] = v
    return Q

def spectral_projector_to_kernel(A, k, M=None):
    if M is None:
        vals, vecs = np.linalg.eigh(A)
        idx = np.argsort(vals)[:k]
        B = vecs[:, idx]
        return m_projector(B, M=None), vals
    else:
        mvals, mvecs = np.linalg.eigh(M)
        Minvhalf = (mvecs * (1.0 / np.sqrt(mvals + EPS))) @ mvecs.T
        Sym = Minvhalf @ A @ Minvhalf
        vals, U = np.linalg.eigh(Sym)
        idx = np.argsort(vals)[:k]
        U_k = U[:, idx]
        V_k = Minvhalf @ U_k
        return m_projector(V_k, M=M), vals

def op_norm(X):
    return float(np.linalg.norm(X, 2))

def test_alpha_rho_S_basics(verbose=True):
    rng = np.random.default_rng(42)
    E, k = 192, 3
    B = m_orthonormal_basis(rng.normal(size=(E, k)), M=None)
    expected_alpha = np.array([0.6, 0.3, 0.1])
    v = B @ expected_alpha
    alpha = B.T @ v
    rho = (alpha**2) / np.sum(alpha**2)
    S = shannon_entropy(alpha**2)
    expected_rho = expected_alpha**2 / np.sum(expected_alpha**2)
    expected_S = shannon_entropy(expected_alpha**2)
    # FIXED: Replaced '&&' with 'and'
    ok = (np.allclose(alpha, expected_alpha, atol=1e-12) and
          np.allclose(rho, expected_rho, atol=1e-12) and
          abs(S - expected_S) < 1e-12)
    if verbose:
        print("Alpha:", alpha, "expected", expected_alpha)
        print("Rho:", rho, "expected", expected_rho)
        print("Entropy:", S, "expected", expected_S)
        print("Alpha/Rho/S basics:", "PASS" if ok else "FAIL")
    return ok

def test_entropy_edge_cases(verbose=True):
    w = np.array([3.0, 1.5, 0.5])
    S1 = shannon_entropy(w)
    S2 = shannon_entropy(10.0 * w)
    inv = abs(S1 - S2) < 1e-12

    wz = np.array([1.0, 0.0, 0.0])
    Sz = shannon_entropy(wz)
    zero_ok = abs(Sz - 0.0) < 1e-12

    u = np.ones(5)
    peaked = np.array([1.0, 0, 0, 0, 0])
    Su = shannon_entropy(u)
    Sp = shannon_entropy(peaked)
    order_ok = Su > Sp

    ok = inv and zero_ok and order_ok
    if verbose:
        print("Entropy invariance (scale):", "PASS" if inv else "FAIL", "| S1=", S1, "S2=", S2)
        print("Entropy zeros safe:", "PASS" if zero_ok else "FAIL", "| S([1,0,0])=", Sz)
        print("Uniform > peaked:", "PASS" if order_ok else "FAIL", "| Su=", Su, "Sp=", Sp)
        print("Entropy edge cases:", "PASS" if ok else "FAIL")
    return ok

def test_projector_idempotence(verbose=True):
    rng = np.random.default_rng(0)
    n, k = 50, 3
    X = rng.normal(size=(n, k))
    B = m_orthonormal_basis(X, M=None)
    P = m_projector(B, M=None)

    sym_ok = np.allclose(P, P.T, atol=1e-12)
    idem_ok = np.allclose(P @ P, P, atol=1e-12)

    r = rng.normal(size=(n,))
    y = (np.eye(n) - P) @ r
    orth_ok = np.linalg.norm(P @ y) < 1e-10

    ok = sym_ok and idem_ok and orth_ok
    if verbose:
        print("P symmetric:", "PASS" if sym_ok else "FAIL")
        print("P idempotent:", "PASS" if idem_ok else "FAIL")
        print("P annihilates complement:", "PASS" if orth_ok else "FAIL", "| ||P y||=", np.linalg.norm(P @ y))
        print("Projector idempotence:", "PASS" if ok else "FAIL")
    return ok

def test_davis_kahan_budget(verbose=True):
    rng = np.random.default_rng(1)
    n, k = 40, 3
    diag_vals = np.arange(n, dtype=float)
    diag_vals[:k] = 0.0
    diag_vals[k] = 1.0
    for i in range(k+1, n):
        diag_vals[i] = 1.0 + (i - k) * 0.1
    A = np.diag(diag_vals)
    gap = 1.0
    S_rand = rng.normal(size=(n, n))
    S = 0.5 * (S_rand + S_rand.T)
    Snorm = op_norm(S)
    delta = 1e-3
    A_prime = A + (delta / (Snorm + EPS)) * S
    E_norm = op_norm(A_prime - A)

    P_A, vals_A = spectral_projector_to_kernel(A, k, M=None)
    P_Ap, vals_Ap = spectral_projector_to_kernel(A_prime, k, M=None)

    drift = op_norm(P_A - P_Ap)
    bound = E_norm / gap + 5e-6
    ok = drift <= bound
    if verbose:
        print(f"DK test: ||P - P'|| = {drift:.6e} ; ||E||/gap = {E_norm:.6e} ; PASS={ok}")
    return ok

def run_all():
    results = []
    results.append(("alpha_rho_S_basics", test_alpha_rho_S_basics()))
    results.append(("entropy_edge_cases", test_entropy_edge_cases()))
    results.append(("projector_idempotence", test_projector_idempotence()))
    results.append(("davis_kahan_budget", test_davis_kahan_budget()))
    print("\nSummary")
    for name, ok in results:
        print(f"  {name:>24}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    return all_ok

all_ok = run_all()
print("\nALL TESTS:", "PASS ✅" if all_ok else "FAIL ❌")