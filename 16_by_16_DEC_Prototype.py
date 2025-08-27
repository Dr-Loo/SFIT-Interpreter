# DEC on T^2 (torus) — 16x16 -> 32x32 prototype with periodic BCs
# Builds d0, d1, Hodge stars, Laplacian on 1-forms,
# extracts harmonic basis (dim=2), and performs one refinement step.

import numpy as np

# --- indexing helpers (periodic) ---
def idx_node(i, j, Nx, Ny):  return (i % Nx) * Ny + (j % Ny)
def idx_xedge(i, j, Nx, Ny): return (i % Nx) * Ny + (j % Ny)               # first block
def idx_yedge(i, j, Nx, Ny): return Nx*Ny + (i % Nx) * Ny + (j % Ny)        # second block
def idx_face(i, j, Nx, Ny):  return (i % Nx) * Ny + (j % Ny)

# --- build DEC operators and Hodge stars on a uniform T^2 grid ---
def build_dec_mats(Nx, Ny):
    N0, N1, N2 = Nx*Ny, 2*Nx*Ny, Nx*Ny
    d0 = np.zeros((N1, N0), float)
    # x-edges: f(head) - f(tail)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_xedge(i, j, Nx, Ny)
            d0[r, idx_node(i+1,j,Nx,Ny)] += 1.0
            d0[r, idx_node(i,  j,Nx,Ny)] -= 1.0
    # y-edges
    for i in range(Nx):
        for j in range(Ny):
            r = idx_yedge(i, j, Nx, Ny)
            d0[r, idx_node(i, j+1,Nx,Ny)] += 1.0
            d0[r, idx_node(i, j,  Nx,Ny)] -= 1.0

    d1 = np.zeros((N2, N1), float)
    # face boundary: +x (bottom), +y (right), -x (top), -y (left)
    for i in range(Nx):
        for j in range(Ny):
            r = idx_face(i, j, Nx, Ny)
            d1[r, idx_xedge(i,  j,  Nx,Ny)] += 1.0
            d1[r, idx_yedge(i+1,j,  Nx,Ny)] += 1.0
            d1[r, idx_xedge(i,  j+1,Nx,Ny)] -= 1.0
            d1[r, idx_yedge(i,  j,  Nx,Ny)] -= 1.0

    dx, dy = 1.0/Nx, 1.0/Ny
    M0 = (dx*dy) * np.eye(N0)
    # 1-forms: x-edges weight ~ dy/dx ; y-edges weight ~ dx/dy
    w_x, w_y = dy/dx, dx/dy
    M1 = np.zeros((N1, N1))
    np.fill_diagonal(M1[:Nx*Ny, :Nx*Ny], w_x)
    np.fill_diagonal(M1[Nx*Ny:, Nx*Ny:], w_y)
    M2 = (1.0/(dx*dy)) * np.eye(N2)
    return d0, d1, M0, M1, M2

# --- Laplacian on 1-forms (self-adjoint in the M1 inner product) ---
def laplacian_1form(d0, d1, M0, M1, M2):
    # ∆1 = d0 d0* + d1* d1, with adjoints: d0* = M0^{-1} d0^T M1 ; d1* = M1^{-1} d1^T M2
    term1 = d0 @ (np.linalg.inv(M0) @ (d0.T @ M1))
    term2 = (np.linalg.inv(M1) @ (d1.T @ (M2 @ d1)))
    return term1 + term2

# --- generalized eigenpairs via M1^{-1/2} transform; returns M1-orthonormal basis ---
def generalized_eigs(L, M1, nev=4, eps=1e-12):
    diag_M1 = np.diag(M1)
    S = np.diag(1.0 / np.sqrt(diag_M1))       # M1^{-1/2}
    B = S @ L @ S                              # symmetric standard form
    w, U = np.linalg.eigh(B)                   # sort ascending
    idx = np.argsort(w); w, U = w[idx], U[:,idx]
    X = S @ U                                  # back-map eigenvectors
    # M1-orthonormalize the first 'nev' via Gram–Schmidt
    def m1_dot(a,b): return a.T @ (M1 @ b)
    Q = []
    for j in range(min(nev, X.shape[1])):
        v = X[:,j].copy()
        for q in Q:
            v = v - q * m1_dot(q, v)
        nrm = np.sqrt(max(m1_dot(v, v), eps))
        Q.append(v / nrm)
    return w[:nev], np.stack(Q, axis=1)        # eigenvalues, M1-orthonormal basis columns

# --- prolongation/restriction on edges:  (Nx,Ny) → (2Nx,2Ny) ---
def build_prolong_restrict_edges(Nx, Ny):
    Nx2, Ny2 = 2*Nx, 2*Ny
    Ec, Ef = 2*Nx*Ny, 2*Nx2*Ny2
    P = np.zeros((Ef, Ec))
    # x-edges: coarse (i,j) -> fine (2i,2j) and (2i+1,2j)
    for i in range(Nx):
        for j in range(Ny):
            rc = idx_xedge(i,j,Nx,Ny)
            P[idx_xedge(2*i,  2*j, Nx2,Ny2), rc] = 1.0
            P[idx_xedge(2*i+1,2*j, Nx2,Ny2), rc] = 1.0
    # y-edges: coarse (i,j) -> fine (2i,2j) and (2i,2j+1)
    for i in range(Nx):
        for j in range(Ny):
            rc = idx_yedge(i,j,Nx,Ny)
            P[idx_yedge(2*i,2*j,  Nx2,Ny2), rc] = 1.0
            P[idx_yedge(2*i,2*j+1,Nx2,Ny2), rc] = 1.0
    R = 0.5 * P.T                               # average back
    return P, R

def projector_from_basis(B, M1):
    # columns of B are M1-orthonormal → Π = B B^T M1
    return B @ (B.T @ M1)

def symbolic_weights(v, B, M1, eps=1e-15):
    alpha = B.T @ (M1 @ v)
    w = np.abs(alpha)**2
    rho = w / (w.sum() + eps)
    return alpha, rho

def entropy_SU(rho, eps=1e-15):
    r = np.clip(rho, eps, 1.0)
    return float(-np.sum(r * np.log(r)))

# --- build coarse level (16x16) ---
Nx, Ny = 16, 16
d0_c, d1_c, M0_c, M1_c, M2_c = build_dec_mats(Nx, Ny)
L_c = laplacian_1form(d0_c, d1_c, M0_c, M1_c, M2_c)
w_c, B_c = generalized_eigs(L_c, M1_c, nev=6)    # expect two ~0 eigenvalues on T^2

# --- build fine level (32x32) ---
d0_f, d1_f, M0_f, M1_f, M2_f = build_dec_mats(2*Nx, 2*Ny)
L_f = laplacian_1form(d0_f, d1_f, M0_f, M1_f, M2_f)
w_f, B_f = generalized_eigs(L_f, M1_f, nev=6)

# harmonic bases (2D)
k = 2
B_c_h = B_c[:, :k]
B_f_h = B_f[:, :k]
Pi_c  = projector_from_basis(B_c_h, M1_c)
Pi_f  = projector_from_basis(B_f_h, M1_f)

# prolongation/restriction
P, R = build_prolong_restrict_edges(Nx, Ny)

# encode a symbolic state on coarse level
theta = 0.37
v_c = np.cos(theta)*B_c_h[:,0] + np.sin(theta)*B_c_h[:,1]
v_c = v_c / np.sqrt(v_c.T @ (M1_c @ v_c))       # M1_c-normalize

# push to fine, add orthogonal noise, project back
v_f_raw = P @ v_c
rng = np.random.default_rng(0)
eta = rng.normal(size=v_f_raw.shape)
eta_orth = eta - (Pi_f @ eta)                   # remove harmonic part
eta_orth /= np.sqrt(eta_orth.T @ (M1_f @ eta_orth)) + 1e-15
v_f = v_f_raw + 0.05 * eta_orth                 # SNR ~ 26 dB

v_c_rec = R @ (Pi_f @ v_f)

# cycle weights and entropy (coarse target vs recovered)
_, rho_c     = symbolic_weights(v_c,     B_c_h, M1_c)
_, rho_c_rec = symbolic_weights(v_c_rec, B_c_h, M1_c)
S_c     = entropy_SU(rho_c)
S_c_rec = entropy_SU(rho_c_rec)

# projector drift (two proxies)
Drift = Pi_c - R @ Pi_f @ P
fro = np.linalg.norm(Drift, 'fro')
sqrtM1   = np.diag(np.sqrt(np.diag(M1_c)))
invSqrtM = np.diag(1.0/np.sqrt(np.diag(M1_c)))
fro_M = np.linalg.norm(sqrtM1 @ Drift @ invSqrtM, 'fro')

print("=== DEC on T^2 (16x16 -> 32x32) ===")
print("Smallest coarse eigenvalues (∆1 vs M1):", np.round(w_c[:6], 12))
print("Smallest fine   eigenvalues (∆1 vs M1):", np.round(w_f[:6], 12))
print("\nCycle weights ρ (coarse, target):", np.round(rho_c, 6))
print("Cycle weights ρ (coarse, after refine+project):", np.round(rho_c_rec, 6))
print(f"Entropy S_U  (target) = {S_c:.6f}")
print(f"Entropy S_U  (recovered) = {S_c_rec:.6f}")
print("\nProjector drift proxies:")
print(f"‖Π_c - R Π_f P‖_F = {fro:.3e}")
print(f"‖M1^{1/2}(Π_c - R Π_f P)M1^{-1/2}‖_F = {fro_M:.3e}")
