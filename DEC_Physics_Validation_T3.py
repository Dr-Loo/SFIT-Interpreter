# DEC_Physics_Validation_T3.py
# Physics checks on T^3 (periodic) for 1-forms/0-forms using DEC:
# 1) Hodge decomposition   ω = dα + δβ + γ, with γ ∈ H^1
# 2) Poisson solve         Δ0 φ = ρ (zero-mean RHS; handle nullspace with pseudoinverse)
# 3) Eigenvalue convergence of Δ1 under refinement (8x6x5 -> 16x12x10 by default)

import numpy as np

# -------------------- Indexing helpers (periodic grid) --------------------
def idx_node(i, j, k, Nx, Ny, Nz):
    return ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_xedge(i, j, k, Nx, Ny, Nz):
    # x-edges block 0
    return ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_yedge(i, j, k, Nx, Ny, Nz):
    # y-edges block 1
    return Nx * Ny * Nz + ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_zedge(i, j, k, Nx, Ny, Nz):
    # z-edges block 2
    return 2 * Nx * Ny * Nz + ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_xyface(i, j, k, Nx, Ny, Nz):
    # xy faces block 0
    return ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_yzface(i, j, k, Nx, Ny, Nz):
    # yz faces block 1
    return Nx * Ny * Nz + ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

def idx_zxface(i, j, k, Nx, Ny, Nz):
    # zx faces block 2
    return 2 * Nx * Ny * Nz + ((i % Nx) * Ny + (j % Ny)) * Nz + (k % Nz)

# -------------------- Build DEC operators and Hodge stars on T^3 --------------------
def build_dec_mats_3d(Nx, Ny, Nz):
    N0 = Nx * Ny * Nz
    N1 = 3 * Nx * Ny * Nz
    N2 = 3 * Nx * Ny * Nz

    d0 = np.zeros((N1, N0), dtype=float)

    # x-edges: f(i+1,j,k) - f(i,j,k)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_xedge(i, j, k, Nx, Ny, Nz)
                d0[r, idx_node(i+1, j,   k,   Nx, Ny, Nz)] += 1.0
                d0[r, idx_node(i,   j,   k,   Nx, Ny, Nz)] -= 1.0
    # y-edges
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_yedge(i, j, k, Nx, Ny, Nz)
                d0[r, idx_node(i,   j+1, k,   Nx, Ny, Nz)] += 1.0
                d0[r, idx_node(i,   j,   k,   Nx, Ny, Nz)] -= 1.0
    # z-edges
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_zedge(i, j, k, Nx, Ny, Nz)
                d0[r, idx_node(i,   j,   k+1, Nx, Ny, Nz)] += 1.0
                d0[r, idx_node(i,   j,   k,   Nx, Ny, Nz)] -= 1.0

    d1 = np.zeros((N2, N1), dtype=float)

    # xy faces (normal +z): +Ex(i,j,k) +Ey(i+1,j,k) -Ex(i,j+1,k) -Ey(i,j,k)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_xyface(i, j, k, Nx, Ny, Nz)
                d1[r, idx_xedge(i,   j,   k, Nx, Ny, Nz)] += 1.0
                d1[r, idx_yedge(i+1, j,   k, Nx, Ny, Nz)] += 1.0
                d1[r, idx_xedge(i,   j+1, k, Nx, Ny, Nz)] -= 1.0
                d1[r, idx_yedge(i,   j,   k, Nx, Ny, Nz)] -= 1.0

    # yz faces (normal +x): +Ey(i,j,k) +Ez(i,j+1,k) -Ey(i,j,k+1) -Ez(i,j,k)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_yzface(i, j, k, Nx, Ny, Nz)
                d1[r, idx_yedge(i, j,   k,   Nx, Ny, Nz)] += 1.0
                d1[r, idx_zedge(i, j+1, k,   Nx, Ny, Nz)] += 1.0
                d1[r, idx_yedge(i, j,   k+1, Nx, Ny, Nz)] -= 1.0
                d1[r, idx_zedge(i, j,   k,   Nx, Ny, Nz)] -= 1.0

    # zx faces (normal +y): +Ez(i,j,k) +Ex(i,j,k+1) -Ez(i+1,j,k) -Ex(i,j,k)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_zxface(i, j, k, Nx, Ny, Nz)
                d1[r, idx_zedge(i,   j,   k,   Nx, Ny, Nz)] += 1.0
                d1[r, idx_xedge(i,   j,   k+1, Nx, Ny, Nz)] += 1.0
                d1[r, idx_zedge(i+1, j,   k,   Nx, Ny, Nz)] -= 1.0
                d1[r, idx_xedge(i,   j,   k,   Nx, Ny, Nz)] -= 1.0

    dx, dy, dz = 1.0 / Nx, 1.0 / Ny, 1.0 / Nz

    # Mass matrices (Hodge stars)
    # 0-forms: dual volume
    M0 = (dx * dy * dz) * np.eye(N0)

    # 1-forms: block diag weights
    M1 = np.zeros((N1, N1))
    w_ex = (dy * dz) / dx
    w_ey = (dx * dz) / dy
    w_ez = (dx * dy) / dz
    np.fill_diagonal(M1[0: Nx*Ny*Nz, 0: Nx*Ny*Nz], w_ex)
    np.fill_diagonal(M1[Nx*Ny*Nz: 2*Nx*Ny*Nz, Nx*Ny*Nz: 2*Nx*Ny*Nz], w_ey)
    np.fill_diagonal(M1[2*Nx*Ny*Nz:, 2*Nx*Ny*Nz:], w_ez)

    # 2-forms: block diag weights
    M2 = np.zeros((N2, N2))
    w_fxy = dz / (dx * dy)
    w_fyz = dx / (dy * dz)
    w_fzx = dy / (dz * dx)
    np.fill_diagonal(M2[0: Nx*Ny*Nz, 0: Nx*Ny*Nz], w_fxy)
    np.fill_diagonal(M2[Nx*Ny*Nz: 2*Nx*Ny*Nz, Nx*Ny*Nz: 2*Nx*Ny*Nz], w_fyz)
    np.fill_diagonal(M2[2*Nx*Ny*Nz:, 2*Nx*Ny*Nz:], w_fzx)

    # 3-forms (unused here) would be 1/(dx*dy*dz) * I

    return d0, d1, M0, M1, M2

# -------------------- Laplacian on 1-forms --------------------
def laplacian_1form(d0, d1, M0, M1, M2):
    # ∆1 = d0 d0* + d1* d1, with adjoints:
    # d0* = M0^{-1} d0^T M1 ;  d1* = M1^{-1} d1^T M2
    term1 = d0 @ (np.linalg.inv(M0) @ (d0.T @ M1))
    term2 = (np.linalg.inv(M1) @ (d1.T @ (M2 @ d1)))
    return term1 + term2

# -------------------- Generalized eigensolver via M1^{-1/2} --------------------
def generalized_eigs(L, M1, nev=12, eps=1e-12):
    diag_M1 = np.diag(M1)
    S = np.diag(1.0 / np.sqrt(diag_M1 + 0.0))  # M1^{-1/2}
    B = S @ L @ S                               # symmetric
    w, U = np.linalg.eigh(B)                    # ascending
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

# -------------------- Hodge decomposition of a 1-form --------------------
def hodge_decomposition_1form(omega, d0, d1, M0, M1, M2):
    """
    ω ∈ C^1. Compute exact dα, coexact δβ, harmonic γ via:
      - exact:    (d0^T M1 d0) α = d0^T M1 ω     ⇒ dα = d0 α
      - coexact:  (d1 d1*) x   = d1 ω            ⇒ δβ = d1* x
    """
    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    d1_star = np.linalg.inv(M1) @ (d1.T @ M2)

    # Exact part
    A0 = d0.T @ (M1 @ d0)             # singular (constants)
    b0 = d0.T @ (M1 @ omega)
    alpha = np.linalg.pinv(A0) @ b0
    d_alpha = d0 @ alpha

    # Coexact part
    L2 = d1 @ d1_star                  # singular along H^2
    rhs2 = d1 @ omega
    x2 = np.linalg.pinv(L2) @ rhs2
    delta_beta = d1_star @ x2

    # Harmonic part
    gamma = omega - d_alpha - delta_beta
    return d_alpha, delta_beta, gamma

# -------------------- Norms and inner products --------------------
def norm_M(v, M):
    return float(np.sqrt(max(v.T @ (M @ v), 0.0)))

def inner_M(u, v, M):
    return float(u.T @ (M @ v))

# -------------------- Poisson solve on 0-forms: Δ0 φ = ρ --------------------
def poisson_0form_solve(d0, M0, M1, rho):
    """
    Δ0 = d0* d0 with d0* = M0^{-1} d0^T M1  ⇒  Δ0 = M0^{-1} (d0^T M1 d0)
    Solve (d0^T M1 d0) φ = M0 ρ0 with zero-mean ρ0; pseudoinverse handles nullspace.
    """
    ones = np.ones(M0.shape[0])
    vol  = float(ones.T @ (M0 @ ones))
    mean = float(ones.T @ (M0 @ rho)) / vol
    rho0 = rho - mean * ones

    A = d0.T @ (M1 @ d0)
    rhs = M0 @ rho0
    phi = np.linalg.pinv(A) @ rhs    # minimal-norm solution; φ defined up to constants

    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    resid = (d0_star @ (d0 @ phi)) - rho0
    return phi, rho0, resid

# -------------------- Eigenvalue convergence of Δ1 under refinement --------------------
def eigen_convergence_3d(Nx, Ny, Nz, nev=12):
    d0_c, d1_c, M0_c, M1_c, M2_c = build_dec_mats_3d(Nx, Ny, Nz)
    L1_c = laplacian_1form(d0_c, d1_c, M0_c, M1_c, M2_c)
    w_c, _ = generalized_eigs(L1_c, M1_c, nev=nev)

    d0_f, d1_f, M0_f, M1_f, M2_f = build_dec_mats_3d(2*Nx, 2*Ny, 2*Nz)
    L1_f = laplacian_1form(d0_f, d1_f, M0_f, M1_f, M2_f)
    w_f, _ = generalized_eigs(L1_f, M1_f, nev=nev)

    # Drop the three zero modes (dim H^1 = 3 on T^3)
    nz_c = w_c[3:nev]
    nz_f = w_f[3:nev]
    k = min(len(nz_c), len(nz_f))
    rel_change = np.abs(nz_f[:k] - nz_c[:k]) / np.maximum(1e-15, np.abs(nz_f[:k]))
    return w_c, w_f, rel_change

# -------------------- Run the three physics validations --------------------
def run_physics_validation_3d(Nx=8, Ny=6, Nz=5, seed=0):
    print(f"=== DEC Physics Validation on T^3 ({Nx}x{Ny}x{Nz} -> {2*Nx}x{2*Ny}x{2*Nz}) ===")
    rng = np.random.default_rng(seed)

    # Build coarse DEC
    d0, d1, M0, M1, M2 = build_dec_mats_3d(Nx, Ny, Nz)

    # (1) Hodge decomposition test on random 1-form ω
    omega = rng.normal(size=(3*Nx*Ny*Nz,))
    d_alpha, delta_beta, gamma = hodge_decomposition_1form(omega, d0, d1, M0, M1, M2)

    # Checks: div(gamma) ≈ 0; curl(gamma) ≈ 0; orthogonality; energy split
    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    div_gamma  = d0_star @ gamma              # 0-form
    curl_gamma = d1 @ gamma                   # 2-form

    norm_omega = norm_M(omega,     M1)
    norm_ex    = norm_M(d_alpha,   M1)
    norm_coex  = norm_M(delta_beta, M1)
    norm_harm  = norm_M(gamma,     M1)
    energy_err = abs(norm_omega**2 - (norm_ex**2 + norm_coex**2 + norm_harm**2))

    ort_ex_h   = abs(inner_M(d_alpha,   gamma, M1))
    ort_co_h   = abs(inner_M(delta_beta, gamma, M1))
    div_norm   = norm_M(div_gamma,  M0)
    curl_norm  = norm_M(curl_gamma, M2)

    print("\n-- Hodge decomposition (1-form) --")
    print(f"‖ω‖_M1^2 ≈ {norm_omega**2:.6e} ; split: exact {norm_ex**2:.6e}  coexact {norm_coex**2:.6e}  harmonic {norm_harm**2:.6e}")
    print(f"Energy closure |‖ω‖^2 − sum| = {energy_err:.3e}")
    print(f"Orthogonality ⟨dα,γ⟩ = {ort_ex_h:.3e} ; ⟨δβ,γ⟩ = {ort_co_h:.3e}")
    print(f"Constraints ‖div γ‖_M0 = {div_norm:.3e} ; ‖curl γ‖_M2 = {curl_norm:.3e}")

    # (2) Poisson solve Δ0 φ = ρ with zero-mean RHS (coarse grid)
    rho = rng.normal(size=(Nx*Ny*Nz,))
    phi, rho0, resid = poisson_0form_solve(d0, M0, M1, rho)
    res_norm = norm_M(resid, M0)
    rhs_norm = norm_M(rho0, M0)
    ones = np.ones(M0.shape[0])
    mean_phi = float(ones.T @ (M0 @ phi)) / float(ones.T @ (M0 @ ones))
    print("\n-- Poisson 0-form (Δ0 φ = ρ) --")
    print(f"‖ρ0‖_M0 = {rhs_norm:.6e} ; ‖Δ0 φ − ρ0‖_M0 = {res_norm:.3e} (expect ≈ numerical roundoff)")
    print(f"Gauge check: mean(φ) ≈ {mean_phi:.3e} (free constant)")

    # (3) Eigenvalue convergence for Δ1 (coarse vs fine)
    w_c, w_f, rel_change = eigen_convergence_3d(Nx, Ny, Nz, nev=12)
    print("\n-- Eigenvalue convergence (Δ1) --")
    print("Coarse eigenvalues (first 12):", np.round(w_c, 6))
    print("Fine   eigenvalues (first 12):", np.round(w_f, 6))
    print("Rel. change of first nonzeros:", np.round(rel_change[:6], 6))

if __name__ == "__main__":
    run_physics_validation_3d(Nx=8, Ny=6, Nz=5, seed=0)
