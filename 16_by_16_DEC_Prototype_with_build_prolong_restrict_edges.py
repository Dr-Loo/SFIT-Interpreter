# 16_by_16_DEC_Prototype_with_build_prolong_restrict_edges.py
# DEC on T^2 (torus) — 16x16 -> 32x32 with periodic BCs
# Uses M-adjoint restriction R = M1_c^{-1} P^T M1_f to eliminate artificial projector drift.
# Reports: eigenvalues, cycle weights ρ, entropy S_U, projector drift, Davis–Kahan budget, principal angles.

import numpy as np

# ---------- Indexing helpers (periodic grid) ----------
def idx_node(i, j, Nx, Ny):
    return (i % Nx) * Ny + (j % Ny)

def idx_xedge(i, j, Nx, Ny):
    # x-edge from (i,j) -> (i+1,j). Stored in first block [0 : Nx*Ny)
    return (i % Nx) * Ny + (j % Ny)

def idx_yedge(i, j, Nx, Ny):
    # y-edge from (i,j) -> (i,j+1). Stored in second block [Nx*Ny : 2*Nx*Ny)
    return Nx * Ny + (i % Nx) * Ny + (j % Ny)

def idx_face(i, j, Nx, Ny):
    return (i % Nx) * Ny + (j % Ny)

# ---------- Build DEC operators and Hodge stars ----------
def build_dec_mats(Nx, Ny):
    N0 = Nx * Ny
    N1 = 2 * Nx * Ny
    N2 = Nx * Ny

    d0 = np.zeros((N1, N0), dtype=float)
    # x-edges: f(head) - f(tail)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_xedge(i, j, Nx, Ny)
            d0[r, idx_node(i+1, j, Nx, Ny)] += 1.0
            d0[r, idx_node(i,   j, Nx, Ny)] -= 1.0
    # y-edges
    for i in range(Nx):
        for j in range(Ny):
            r = idx_yedge(i, j, Nx, Ny)
            d0[r, idx_node(i, j+1, Nx, Ny)] += 1.0
            d0[r, idx_node(i, j,   Nx, Ny)] -= 1.0

    d1 = np.zeros((N2, N1), dtype=float)
    # face boundary: +x (bottom), +y (right), -x (top), -y (left)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_face(i, j, Nx, Ny)
            d1[r, idx_xedge(i,   j,   Nx, Ny)] += 1.0
            d1[r, idx_yedge(i+1, j,   Nx, Ny)] += 1.0
            d1[r, idx_xedge(i,   j+1, Nx, Ny)] -= 1.0
            d1[r, idx_yedge(i,   j,   Nx, Ny)] -= 1.0

    dx, dy = 1.0 / Nx, 1.0 / Ny

    # Mass matrices (Hodge stars)
    M0 = (dx * dy) * np.eye(N0)
    # 1-forms: x-edges weight ~ dy/dx ; y-edges weight ~ dx/dy
    w_x, w_y = dy / dx, dx / dy
    M1 = np.zeros((N1, N1))
    np.fill_diagonal(M1[:Nx*Ny, :Nx*Ny], w_x)
    np.fill_diagonal(M1[Nx*Ny:, Nx*Ny:], w_y)
    # 2-forms: inverse area
    M2 = (1.0 / (dx * dy)) * np.eye(N2)

    return d0, d1, M0, M1, M2

# ---------- Laplacian on 1-forms ----------
def laplacian_1form(d0, d1, M0, M1, M2):
    # ∆1 = d0 d0* + d1* d1, with adjoints: d0* = M0^{-1} d0^T M1 ; d1* = M1^{-1} d1^T M2
    term1 = d0 @ (np.linalg.inv(M0) @ (d0.T @ M1))
    term2 = (np.linalg.inv(M1) @ (d1.T @ (M2 @ d1)))
    return term1 + term2

# ---------- Generalized eigenpairs via M1^{-1/2} transform ----------
def generalized_eigs(L, M1, nev=6, eps=1e-12):
    diag_M1 = np.diag(M1)
    S = np.diag(1.0 / np.sqrt(diag_M1 + 0.0))  # M1^{-1/2}
    B = S @ L @ S                               # symmetric
    w, U = np.linalg.eigh(B)                    # sorted ascending
    idx = np.argsort(w)
    w = w[idx]
    U = U[:, idx]
    X = S @ U                                   # back-map

    # M1-orthonormalize first 'nev' columns (Gram–Schmidt in M1 inner product)
    def m1_dot(a, b): return float(a.T @ (M1 @ b))
    Q = []
    for j in range(min(nev, X.shape[1])):
        v = X[:, j].copy()
        for q in Q:
            v = v - q * m1_dot(q, v)
        nrm = np.sqrt(max(m1_dot(v, v), eps))
        Q.append(v / nrm)
    return w[:nev], np.stack(Q, axis=1)

# ---------- Prolongation/Restriction with M-adjoint ----------
def build_prolong_restrict_edges_M_adj(Nx, Ny, M1_c, M1_f):
    """
    Prolong P : edges_(Nx,Ny) -> edges_(2Nx,2Ny) via replication of coarse edges into two fine children.
    Restrict R : M-adjoint  R = M1_c^{-1} P^T M1_f  (eliminates artificial drift).
    """
    Nx2, Ny2 = 2*Nx, 2*Ny
    Ec, Ef = 2*Nx*Ny, 2*Nx2*Ny2
    P = np.zeros((Ef, Ec))
    # x-edges: coarse (i,j) -> fine (2i,2j) and (2i+1,2j)
    for i in range(Nx):
        for j in range(Ny):
            rc = idx_xedge(i, j, Nx, Ny)
            P[idx_xedge(2*i,   2*j,   Nx2, Ny2), rc] = 1.0
            P[idx_xedge(2*i+1, 2*j,   Nx2, Ny2), rc] = 1.0
    # y-edges: coarse (i,j) -> fine (2i,2j) and (2i,2j+1)
    for i in range(Nx):
        for j in range(Ny):
            rc = idx_yedge(i, j, Nx, Ny)
            P[idx_yedge(2*i,   2*j,   Nx2, Ny2), rc] = 1.0
            P[idx_yedge(2*i,   2*j+1, Nx2, Ny2), rc] = 1.0
    R = np.linalg.inv(M1_c) @ (P.T @ M1_f)
    return P, R

# ---- Compatibility wrapper (so your old call still works) ----
def build_prolong_restrict_edges(Nx, Ny):
    # Rebuild mass matrices internally so the signature matches your old code.
    _, _, _, M1_c, _ = build_dec_mats(Nx, Ny)
    _, _, _, M1_f, _ = build_dec_mats(2*Nx, 2*Ny)
    return build_prolong_restrict_edges_M_adj(Nx, Ny, M1_c, M1_f)

# ---------- Utilities ----------
def projector_from_basis(B, M1):
    # Columns of B are M1-orthonormal → Π = B B^T M1
    return B @ (B.T @ M1)

def symbolic_weights(v, B, M1, eps=1e-15):
    alpha = B.T @ (M1 @ v)
    w = np.abs(alpha)**2
    rho = w / (w.sum() + eps)
    return alpha, rho

def entropy_SU(rho, eps=1e-15):
    r = np.clip(rho, eps, 1.0)
    return float(-np.sum(r * np.log(r)))

def m1_qr(B, M1, eps=1e-12):
    # M1-orthonormalize columns of B (modified Gram–Schmidt)
    def m1_dot(a, b): return float(a.T @ (M1 @ b))
    Q = []
    for j in range(B.shape[1]):
        v = B[:, j].copy()
        for q in Q:
            v = v - q * m1_dot(q, v)
        nrm = np.sqrt(max(m1_dot(v, v), eps))
        Q.append(v / nrm)
    return np.stack(Q, axis=1)

def principal_angles_M(B1, B2, M1):
    """
    Principal angles between subspaces span(B1) and span(B2) w.r.t. M1 inner product.
    Returns angles (in degrees) sorted ascending.
    """
    Q1 = m1_qr(B1, M1)
    Q2 = m1_qr(B2, M1)
    S = Q1.T @ (M1 @ Q2)              # overlap in M1
    # singular values are cosines of angles
    s = np.linalg.svd(S, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return np.sort(angles)

def op_norm_M(A, M):
    """
    Operator 2-norm in the M inner product: ||A||_M = || M^{1/2} A M^{-1/2} ||_2
    """
    sqrtM   = np.diag(np.sqrt(np.diag(M)))
    invSqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
    B = sqrtM @ A @ invSqrt
    return float(np.linalg.norm(B, 2))

# ---------- Main experiment ----------
def run_experiment(Nx=16, Ny=16, noise_level=0.05, theta=0.37, seed=0):
    # Coarse level
    d0_c, d1_c, M0_c, M1_c, M2_c = build_dec_mats(Nx, Ny)
    L_c = laplacian_1form(d0_c, d1_c, M0_c, M1_c, M2_c)
    w_c, B_c = generalized_eigs(L_c, M1_c, nev=6)

    # Fine level
    d0_f, d1_f, M0_f, M1_f, M2_f = build_dec_mats(2*Nx, 2*Ny)
    L_f = laplacian_1form(d0_f, d1_f, M0_f, M1_f, M2_f)
    w_f, B_f = generalized_eigs(L_f, M1_f, nev=6)

    # Harmonic bases (dim=2 on T^2)
    k = 2
    B_c_h = B_c[:, :k]
    B_f_h = B_f[:, :k]
    Pi_c  = projector_from_basis(B_c_h, M1_c)
    Pi_f  = projector_from_basis(B_f_h, M1_f)

    # Prolong/Restrict (M-adjoint). Use the 2-arg wrapper to match your old call:
    P, R = build_prolong_restrict_edges(Nx, Ny)
    # (If you prefer the explicit version, use:
    #  P, R = build_prolong_restrict_edges_M_adj(Nx, Ny, M1_c, M1_f))

    # Encode a symbolic state on coarse level
    v_c = np.cos(theta) * B_c_h[:, 0] + np.sin(theta) * B_c_h[:, 1]
    v_c = v_c / np.sqrt(float(v_c.T @ (M1_c @ v_c)))  # normalize in M1_c

    # Push to fine, add orthogonal noise, project back
    v_f_raw = P @ v_c
    rng = np.random.default_rng(seed)
    eta = rng.normal(size=v_f_raw.shape)
    eta_orth = eta - (Pi_f @ eta)                              # remove harmonic part
    denom = np.sqrt(float(eta_orth.T @ (M1_f @ eta_orth))) + 1e-15
    eta_orth = noise_level * eta_orth / denom
    v_f = v_f_raw + eta_orth

    v_c_rec = R @ (Pi_f @ v_f)

    # Cycle weights and entropy
    _, rho_c     = symbolic_weights(v_c,     B_c_h, M1_c)
    _, rho_c_rec = symbolic_weights(v_c_rec, B_c_h, M1_c)
    S_c     = entropy_SU(rho_c)
    S_c_rec = entropy_SU(rho_c_rec)

    # Projector drift (M1-operator norm and Frobenius)
    Drift = Pi_c - R @ Pi_f @ P
    drift_opM = op_norm_M(Drift, M1_c)
    drift_fro = np.linalg.norm(Drift, 'fro')

    # Davis–Kahan budget: ||E||_M / gap, where E = R ∆_f P − ∆_c, gap = λ_3 (first nonzero)
    E = R @ L_f @ P - L_c
    gap_c = float(w_c[2])  # eigenvalues: [~0, ~0, gap, ...]
    dk_budget = op_norm_M(E, M1_c) / max(gap_c, 1e-15)

    # Principal angles between H^1_c and R(H^1_f)
    angles_deg = principal_angles_M(B_c_h, R @ B_f_h, M1_c)

    # Report
    print("=== DEC on T^2 (16x16 -> 32x32) with M-adjoint restriction ===")
    print("Smallest coarse eigenvalues (∆1 vs M1):", np.round(w_c[:6], 12))
    print("Smallest fine   eigenvalues (∆1 vs M1):", np.round(w_f[:6], 12))
    print("\nCycle weights ρ (coarse, target):               ", np.round(rho_c, 6))
    print("Cycle weights ρ (coarse, after refine+project):", np.round(rho_c_rec, 6))
    print(f"Entropy S_U  (target)    = {S_c:.6f}")
    print(f"Entropy S_U  (recovered) = {S_c_rec:.6f}")
    print("\nProjector drift norms:")
    print(f"‖Π_c - R Π_f P‖_2,M = {drift_opM:.3e}   (M-weighted spectral norm)")
    print(f"‖Π_c - R Π_f P‖_F   = {drift_fro:.3e}   (plain Frobenius)")
    print("\nDavis–Kahan budget:")
    print(f"gap(∆1_c) = λ_3 ≈ {gap_c:.6f}")
    print(f"‖R ∆_f P − ∆_c‖_2,M / gap ≈ {dk_budget:.3e}   (upper bound on subspace rotation)")
    print("\nPrincipal angles H^1_c vs R(H^1_f) [degrees]:", np.round(angles_deg, 6))

if __name__ == '__main__':
    # On Windows you can double-click or run:  python 16_by_16_DEC_Prototype_with_build_prolong_restrict_edges.py
    run_experiment(Nx=16, Ny=16, noise_level=0.05, theta=0.37, seed=0)
