# DEC_Physics_Validation_T2.py
# Physics checks on T^2 (periodic) for 1-forms/0-forms using DEC:
# 1) Hodge decomposition   ω = dα + δβ + γ, with γ ∈ H^1
# 2) Poisson solve         Δ0 φ = ρ  (zero-mean RHS; handle nullspace with pseudoinverse)
# 3) Eigenvalue convergence of Δ1 under refinement (16x16 -> 32x32)

import numpy as np

# -------------------- Indexing helpers (periodic grid) --------------------
def idx_node(i, j, Nx, Ny):
    return (i % Nx) * Ny + (j % Ny)

def idx_xedge(i, j, Nx, Ny):
    return (i % Nx) * Ny + (j % Ny)  # block 0 : x-edges

def idx_yedge(i, j, Nx, Ny):
    return Nx * Ny + (i % Nx) * Ny + (j % Ny)  # block 1 : y-edges

def idx_face(i, j, Nx, Ny):
    return (i % Nx) * Ny + (j % Ny)

# -------------------- Build DEC operators and Hodge stars --------------------
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

# -------------------- Laplacian on 1-forms --------------------
def laplacian_1form(d0, d1, M0, M1, M2):
    # ∆1 = d0 d0* + d1* d1, with adjoints:
    # d0* = M0^{-1} d0^T M1 ;  d1* = M1^{-1} d1^T M2
    term1 = d0 @ (np.linalg.inv(M0) @ (d0.T @ M1))
    term2 = (np.linalg.inv(M1) @ (d1.T @ (M2 @ d1)))
    return term1 + term2

# -------------------- Generalized eigensolver via M1^{-1/2} --------------------
def generalized_eigs(L, M1, nev=10, eps=1e-12):
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
    Given 1-form ω, compute exact dα, coexact δβ, harmonic γ via
      - exact:    solve (d0^T M1 d0) α = d0^T M1 ω     ⇒ dα = d0 α
      - coexact:  solve (d1 d1*) x   = d1 ω            ⇒ δβ = d1* x
    Use pseudoinverses to handle gauge nullspaces (constants, H^2).
    """
    # adjoints
    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    d1_star = np.linalg.inv(M1) @ (d1.T @ M2)

    # Exact part
    A0 = d0.T @ (M1 @ d0)             # (N0 x N0), singular (constants)
    b0 = d0.T @ (M1 @ omega)
    alpha = np.linalg.pinv(A0) @ b0   # minimal-norm solution (imposes zero-mean gauge)
    d_alpha = d0 @ alpha

    # Coexact part
    L2 = d1 @ d1_star                  # (N2 x N2); singular along H^2
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
    Solve for φ with zero-mean RHS; handle nullspace by pseudoinverse:
      (d0^T M1 d0) φ = M0 ρ0
    """
    # enforce zero-mean (orthogonal to constants) wrt M0
    ones = np.ones(M0.shape[0])
    vol  = float(ones.T @ (M0 @ ones))
    mean = float(ones.T @ (M0 @ rho)) / vol
    rho0 = rho - mean * ones

    A = d0.T @ (M1 @ d0)        # (N0 x N0), singular but okay with pinv on zero-mean RHS
    rhs = M0 @ rho0
    phi = np.linalg.pinv(A) @ rhs    # minimal-norm solution; defined mod constants

    # Residual: Δ0 φ − ρ0 = (M0^{-1} A) φ − ρ0
    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    resid = (d0_star @ (d0 @ phi)) - rho0
    return phi, rho0, resid

# -------------------- Eigenvalue convergence of Δ1 --------------------
def eigen_convergence(Nx, Ny, nev=10):
    d0_c, d1_c, M0_c, M1_c, M2_c = build_dec_mats(Nx, Ny)
    L1_c = laplacian_1form(d0_c, d1_c, M0_c, M1_c, M2_c)
    w_c, _ = generalized_eigs(L1_c, M1_c, nev=nev)

    d0_f, d1_f, M0_f, M1_f, M2_f = build_dec_mats(2*Nx, 2*Ny)
    L1_f = laplacian_1form(d0_f, d1_f, M0_f, M1_f, M2_f)
    w_f, _ = generalized_eigs(L1_f, M1_f, nev=nev)

    # Drop the zero modes (dim H^1 = 2 on T^2)
    nz_c = w_c[2:nev]
    nz_f = w_f[2:nev]
    # Compare first few nonzeros
    k = min(len(nz_c), len(nz_f))
    rel_change = np.abs(nz_f[:k] - nz_c[:k]) / np.maximum(1e-15, np.abs(nz_f[:k]))
    return w_c, w_f, rel_change

# -------------------- Run the three physics validations --------------------
def run_physics_validation(Nx=16, Ny=16, seed=0):
    print(f"=== DEC Physics Validation on T^2 ({Nx}x{Ny} -> {2*Nx}x{2*Ny}) ===")
    rng = np.random.default_rng(seed)

    # Build coarse DEC
    d0, d1, M0, M1, M2 = build_dec_mats(Nx, Ny)

    # (1) Hodge decomposition test on random 1-form ω
    omega = rng.normal(size=(2*Nx*Ny,))
    d_alpha, delta_beta, gamma = hodge_decomposition_1form(omega, d0, d1, M0, M1, M2)

    # Checks: div(gamma) ≈ 0; curl(gamma) ≈ 0; orthogonality; energy split
    d0_star = np.linalg.inv(M0) @ (d0.T @ M1)
    d1_star = np.linalg.inv(M1) @ (d1.T @ M2)
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
    rho = rng.normal(size=(Nx*Ny,))
    phi, rho0, resid = poisson_0form_solve(d0, M0, M1, rho)
    res_norm = norm_M(resid, M0)
    rhs_norm = norm_M(rho0, M0)
    print("\n-- Poisson 0-form (Δ0 φ = ρ) --")
    print(f"‖ρ0‖_M0 = {rhs_norm:.6e} ; ‖Δ0 φ − ρ0‖_M0 = {res_norm:.3e} (expect ≈ numerical roundoff)")
    # Optional: report mean(φ) ~ free gauge
    ones = np.ones(M0.shape[0])
    mean_phi = float(ones.T @ (M0 @ phi)) / float(ones.T @ (M0 @ ones))
    print(f"Gauge check: mean(φ) ≈ {mean_phi:.3e} (free constant)")

    # (3) Eigenvalue convergence for Δ1 (coarse vs fine)
    w_c, w_f, rel_change = eigen_convergence(Nx, Ny, nev=10)
    print("\n-- Eigenvalue convergence (Δ1) --")
    print("Coarse eigenvalues (first 10):", np.round(w_c, 6))
    print("Fine   eigenvalues (first 10):", np.round(w_f, 6))
    print("Rel. change of first nonzeros:", np.round(rel_change[:5], 6))

if __name__ == "__main__":
    run_physics_validation(Nx=16, Ny=16, seed=0)
