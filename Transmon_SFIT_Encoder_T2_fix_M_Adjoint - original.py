# Transmon_SFIT_Encoder_T2_fixed.py
# SFIT symbolic encoder on a transmon torus T^2 : (phi, n) ∈ [0,2π) × Z_mod
# Fixes:
#   - M-isometric prolongation (1/sqrt(2) splits) so R P = I in M1 inner product
#   - Optional transported harmonic projector on the fine grid

import numpy as np
from numpy.linalg import eigh

def dk_budget_and_gap(d0c, d1c, M0c, M1c, M2c,
                      d0f, d1f, M0f, M1f, M2f, P, R):
    """Compute coarse gap λ_{k+1} and DK budget ||R Δf P − Δc||_{2,M}/gap."""
    # Generalized Laplacians
    Lc = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Lf = laplacian_1form(d0f, d1f, M0f, M1f, M2f)

    # Coarse gap from generalized eigenproblem Lc x = λ M1c x
    # (use dense eigh on small grids; for big ones, use sparse eigs)
    Mc_half_inv = np.linalg.inv(np.linalg.cholesky(M1c))
    A = Mc_half_inv.T @ Lc @ Mc_half_inv
    vals = np.sort(eigh(A, UPLO='U')[0])
    # T^2/T^3: first k zeros are harmonic; next is the gap:
    #   k=2 for T^2, k=3 for T^3
    k = 2
    gap = float(vals[k])  # λ_{k+1}

    # M-weighted spectral norm of defect E = R Δf P − Δc
    E = R @ (Lf @ P) - Lc
    # ||E||_{2,M} ≡ || M^{-1/2} E M^{-1/2} ||_2
    Em = Mc_half_inv.T @ E @ Mc_half_inv
    # 2-norm upper bound via Frobenius on small problems (ok for monitoring)
    fro = float(np.linalg.norm(Em, 'fro'))
    return gap, fro / max(gap, 1e-15)


import numpy as np

# -------------------- T^2 indexing (periodic) --------------------
def idx_node(i, j, Nx, Ny):   return (i % Nx) * Ny + (j % Ny)
def idx_xedge(i, j, Nx, Ny):  return (i % Nx) * Ny + (j % Ny)                 # block 0
def idx_yedge(i, j, Nx, Ny):  return Nx * Ny + (i % Nx) * Ny + (j % Ny)       # block 1

# -------------------- DEC operators & Hodge stars on T^2 --------------------
def build_dec_mats(Nx, Ny):
    N0, N1, N2 = Nx*Ny, 2*Nx*Ny, Nx*Ny
    d0 = np.zeros((N1, N0), float)
    # x-edges (phi-direction): f(i+1,j) - f(i,j)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_xedge(i,j,Nx,Ny)
            d0[r, idx_node(i+1,j,Nx,Ny)] += 1.0
            d0[r, idx_node(i  ,j,Nx,Ny)] -= 1.0
    # y-edges (n-direction): f(i,j+1) - f(i,j)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_yedge(i,j,Nx,Ny)
            d0[r, idx_node(i,j+1,Nx,Ny)] += 1.0
            d0[r, idx_node(i,j  ,Nx,Ny)] -= 1.0

    d1 = np.zeros((N2, N1), float)
    # face boundary: +Ex(i,j) +Ey(i+1,j) -Ex(i,j+1) -Ey(i,j)
    for i in range(Nx):
        for j in range(Ny):
            r = i*Ny + j
            d1[r, idx_xedge(i  ,j  ,Nx,Ny)] += 1.0
            d1[r, idx_yedge(i+1,j  ,Nx,Ny)] += 1.0
            d1[r, idx_xedge(i  ,j+1,Nx,Ny)] -= 1.0
            d1[r, idx_yedge(i  ,j  ,Nx,Ny)] -= 1.0

    dx, dy = 1.0/Nx, 1.0/Ny
    M0 = (dx*dy) * np.eye(N0)
    M1 = np.zeros((2*Nx*Ny, 2*Nx*Ny))
    # 1-forms: weights dy/dx for x-edges, dx/dy for y-edges
    np.fill_diagonal(M1[:Nx*Ny, :Nx*Ny], dy/dx)
    np.fill_diagonal(M1[Nx*Ny:, Nx*Ny:], dx/dy)
    M2 = (1.0/(dx*dy)) * np.eye(N2)
    return d0, d1, M0, M1, M2

def laplacian_1form(d0, d1, M0, M1, M2):
    term1 = d0 @ (np.linalg.inv(M0) @ (d0.T @ M1))
    term2 = np.linalg.inv(M1) @ (d1.T @ (M2 @ d1))
    return term1 + term2

# -------------------- generalized eigs via M1^{-1/2} --------------------
def generalized_eigs(L, M1, nev=6, eps=1e-12):
    diag = np.diag(M1)
    S = np.diag(1.0/np.sqrt(diag+0.0))
    B = S @ L @ S
    w, U = np.linalg.eigh(B)
    idx = np.argsort(w); w = w[idx]; U = U[:,idx]
    X = S @ U  # back map
    # M1-orthonormalize first nev columns
    def mdot(a,b): return float(a.T @ (M1 @ b))
    Q=[]
    for j in range(min(nev, X.shape[1])):
        v = X[:,j].copy()
        for q in Q: v -= q * mdot(q,v)
        nrm = np.sqrt(max(mdot(v,v), eps)); Q.append(v/nrm)
    return w[:nev], np.stack(Q, axis=1)

def harmonic_basis(M1, L, k):
    w, Q = generalized_eigs(L, M1, nev=max(8, k+4))
    ids = np.argsort(w)[:k]
    return Q[:, ids], w[ids]

def project_harmonic(M1, B, v):
    return B @ (B.T @ (M1 @ v))

def cycle_weights(M1, B, v, eps=1e-18):
    alpha = B.T @ (M1 @ v)
    p = np.abs(alpha)**2
    Z = max(np.sum(p), eps)
    rho = p / Z
    S = -np.sum(rho * (np.log(rho+1e-300)))
    return rho, S, alpha

# -------------------- prolongation / restriction --------------------
def build_prolong_restrict_edges(Nx, Ny, mode="M_isometry"):
    """
    mode = "M_isometry": use 1/sqrt(2) splits so that R = M_c^{-1} P^T M_f gives R P = I on each edge block.
         = "integral":   use 1/2 splits (integral-preserving), R becomes M-adjoint (scales by 1/2 per split).
    """
    N1c = 2*Nx*Ny
    N1f = 2*(2*Nx)*(2*Ny)
    P = np.zeros((N1f, N1c), float)
    s = 1.0/np.sqrt(2.0) if mode == "M_isometry" else 0.5

    # x-edges: split along x (serial)
    for i in range(Nx):
        for j in range(Ny):
            col = idx_xedge(i,j,Nx,Ny)
            r1 = idx_xedge(2*i  , 2*j, 2*Nx, 2*Ny)
            r2 = idx_xedge(2*i+1, 2*j, 2*Nx, 2*Ny)
            P[r1, col] = s
            P[r2, col] = s

    # y-edges: split along y (serial)
    for i in range(Nx):
        for j in range(Ny):
            col = idx_yedge(i,j,Nx,Ny)
            r1 = idx_yedge(2*i, 2*j  , 2*Nx, 2*Ny)
            r2 = idx_yedge(2*i, 2*j+1, 2*Nx, 2*Ny)
            P[r1, col] = s
            P[r2, col] = s

    # M-adjoint restriction
    _, _, _, M1c, _ = build_dec_mats(Nx, Ny)
    _, _, _, M1f, _ = build_dec_mats(2*Nx, 2*Ny)
    R = np.linalg.inv(M1c) @ (P.T @ M1f)
    return P, R

# -------------------- transported projector on fine grid --------------------
def transported_projector(M1f, P, Bc):
    # orthonormalize PBc in M1f
    Y = P @ Bc
    # M1f–Gram–Schmidt
    def mdot(a,b): return float(a.T @ (M1f @ b))
    Q=[]
    for j in range(Y.shape[1]):
        v = Y[:,j].copy()
        for q in Q: v -= q * mdot(q,v)
        nrm = np.sqrt(max(mdot(v,v), 1e-12))
        Q.append(v/nrm)
    Bf_tr = np.stack(Q, axis=1)
    Pif_tr = Bf_tr @ (Bf_tr.T @ M1f)
    return Pif_tr, Bf_tr

# -------------------- noise --------------------
def quasiparticle_noise(M1, dim, scale=0.02, rng=None, B=None):
    if rng is None: rng = np.random.default_rng()
    eta = rng.normal(scale=scale, size=(dim,))
    if B is not None:
        eta = eta - project_harmonic(M1, B, eta)  # strip harmonic part
    return eta

# -------------------- main demo --------------------
def run_transmon_demo(Nx=16, Ny=16, alpha_phi=0.9, alpha_n=0.435,
                      noise=0.03, seed=0,
                      mode="M_isometry",
                      use_transport_projector=True):
    print(f"=== SFIT Transmon Encoder on T^2 ({Nx}x{Ny} -> {2*Nx}x{2*Ny}) mode={mode} ===")
    rng = np.random.default_rng(seed)

    # build DEC
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    d0f, d1f, M0f, M1f, M2f = build_dec_mats(2*Nx, 2*Ny)

    # harmonic bases
    L1c = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    L1f = laplacian_1form(d0f, d1f, M0f, M1f, M2f)
    Bc, _ = harmonic_basis(M1c, L1c, k=2)

    # projector on fine: transported or spectral
    if use_transport_projector:
        P, R = build_prolong_restrict_edges(Nx, Ny, mode=mode)
        Pif, Bf = transported_projector(M1f, P, Bc)
    else:
        P, R = build_prolong_restrict_edges(Nx, Ny, mode=mode)
        Bf, _ = harmonic_basis(M1f, L1f, k=2)
        Pif = Bf @ (Bf.T @ M1f)

    # projectors for drift diagnostics
    Pic = Bc @ (Bc.T @ M1c)

    # encode
    v_c = alpha_phi * Bc[:,0] + alpha_n * Bc[:,1]
    rho_c, S_c, alpha_c = cycle_weights(M1c, Bc, v_c)

    # refine + noise
    v_f_clean = P @ v_c
    eta_f = quasiparticle_noise(M1f, v_f_clean.size, scale=noise, rng=rng, B=Bf)
    v_f = v_f_clean + eta_f

    # recover
    v_c_rec = R @ (Pif @ v_f)
    rho_rec, S_rec, alpha_rec = cycle_weights(M1c, Bc, v_c_rec)

    # errors
    err_M1 = float(np.sqrt((v_c_rec - v_c).T @ (M1c @ (v_c_rec - v_c))))
    # projector drift
    A = Pic - (R @ Pif @ P)
    fro = np.linalg.norm(A, 'fro')
    D = np.sqrt(np.diag(M1c))
    A_M = (D[:,None]*A) / (D[None,:] + 1e-300)
    specM = np.linalg.norm(A_M, 2)

    print("\n-- Harmonic coefficients (coarse) --")
    print("alpha_true  =", np.round(alpha_c, 6))
    print("rho_true    =", np.round(rho_c, 6), "; S_U =", f"{S_c:.6f}")

    print("\n-- After refine + project + restrict (recovered) --")
    print("alpha_rec   =", np.round(alpha_rec, 6))
    print("rho_rec     =", np.round(rho_rec, 6), "; S_U =", f"{S_rec:.6f}")
    print("‖v_rec − v_true‖_M1 =", f"{err_M1:.3e}")

    print("\n-- Projector drift --")
    print("‖Π_c − R Π_f P‖_F     =", f"{fro:.3e}")
    print("‖Π_c − R Π_f P‖_{2,M} =", f"{specM:.3e}")

if __name__ == "__main__":
    run_transmon_demo()
