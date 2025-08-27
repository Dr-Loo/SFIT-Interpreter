# Transmon_SFIT_Encoder_T2_feedback_v9.py
# Stable symbolic feedback with dynamic M-adjoint restriction and conditional projector recalibration

import numpy as np
from numpy.linalg import norm, cholesky, inv
from typing import Tuple

from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats,
    laplacian_1form,
    harmonic_basis,
    cycle_weights,
    build_prolong_restrict_edges,
    transported_projector,
    quasiparticle_noise,
)

# ---------- local helpers (kept here so this file is self-contained) ----------

def apply_metric_bias(M1: np.ndarray, bias_x: float = 1.0, bias_y: float = 1.0,
                      Nx: int = 16, Ny: int = 16) -> np.ndarray:
    """Scale x- and y-edge blocks of M1 by (bias_x, bias_y)."""
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x  # x-edges
    M1b[n:, n:] *= bias_y  # y-edges
    return M1b

def restrict_M_adjoint(P: np.ndarray, M1c: np.ndarray, M1f_current: np.ndarray) -> np.ndarray:
    """M-adjoint restriction for the CURRENT fine metric."""
    return np.linalg.inv(M1c) @ (P.T @ M1f_current)

def quasiparticle_noise_hybrid(M1f: np.ndarray, size: int, scale: float, rng,
                               Bf: np.ndarray, harmonic_frac: float = 0.0) -> np.ndarray:
    """Orthogonal noise + optional harmonic contamination."""
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        # match magnitude to 'scale' (coarse control)
        hn = float(np.sqrt(h_noise.T @ (M1f @ h_noise))) + 1e-15
        h_noise *= (harmonic_frac * scale) / hn
        eta = eta + h_noise
    return eta

def dk_budget_and_gap(d0c: np.ndarray, d1c: np.ndarray, M0c: np.ndarray, M1c: np.ndarray, M2c: np.ndarray,
                      d0f: np.ndarray, d1f: np.ndarray, M0f: np.ndarray, M1f: np.ndarray, M2f: np.ndarray,
                      P: np.ndarray, R: np.ndarray, k: int = 2) -> Tuple[float, float]:
    """Coarse spectral gap (λ_{k+1}) and DK proxy ||RΔfP−Δc||_{2,M}/gap."""
    Lc = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Lf = laplacian_1form(d0f, d1f, M0f, M1f, M2f)
    # eigenvalues of M1c^{-1/2} Lc M1c^{-1/2}
    Mc_half = cholesky(M1c)
    Mc_inv  = inv(Mc_half)
    A = Mc_inv.T @ Lc @ Mc_inv
    vals = np.sort(np.linalg.eigvalsh(A))
    gap = float(vals[k])  # after k harmonic zeros

    # DK budget proxy in M-norm
    E  = R @ (Lf @ P) - Lc
    Em = Mc_inv.T @ E @ Mc_inv
    fro = float(np.linalg.norm(Em, 'fro'))
    DK  = fro / max(gap, 1e-15)
    return gap, DK

# ------------------------------- main routine --------------------------------

def run_v9(Nx=16, Ny=16,
           alpha_phi=0.9, alpha_n=0.435,
           steps=40,
           noise0=0.03, harmonic_noise_frac0=0.20,
           seed=0,
           dk_max=50.0,        # DK proxy threshold to trigger recalibration
           eps_alpha=1e-3,     # coef error threshold
           eps_S=5e-3,         # entropy drift threshold
           bias_bounds=(0.95, 1.05),
           adam_lr=0.02):
    rng = np.random.default_rng(seed)

    # --- coarse reference ---
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    Bc, _ = harmonic_basis(M1c, laplacian_1form(d0c, d1c, M0c, M1c, M2c), k=2)
    v_c   = alpha_phi * Bc[:,0] + alpha_n * Bc[:,1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # --- intergrid base ops ---
    P, _ = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")
    Nx_f, Ny_f = 2*Nx, 2*Ny

    # fine base (unbiased) metric
    d0f, d1f, M0f, M1f_base, M2f = build_dec_mats(Nx_f, Ny_f)

    # biases and projector
    bias_x, bias_y = 1.10, 0.90
    M1f = apply_metric_bias(M1f_base, bias_x, bias_y, Nx_f, Ny_f)
    R   = restrict_M_adjoint(P, M1c, M1f)
    Pif, Bf = transported_projector(M1f, P, Bc)

    # Adam state
    m = np.zeros(2); v = np.zeros(2)
    beta1, beta2, epsAdam = 0.9, 0.999, 1e-8
    bmin = np.array(bias_bounds)
    bmax = np.array(bias_bounds)

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr\t\tgap\t\tDK_budget")
    for it in range(steps):
        # anneal
        noise_now = noise0 * max(0.0, 1.0 - it/(0.8*steps))
        h_harm    = harmonic_noise_frac0 * max(0.0, 1.0 - it/(0.6*steps))

        # current metric + R
        M1f = apply_metric_bias(M1f_base, bias_x, bias_y, Nx_f, Ny_f)
        R   = restrict_M_adjoint(P, M1c, M1f)

        # forward + recover
        v_f_clean = P @ v_c
        eta_f = quasiparticle_noise_hybrid(M1f, v_f_clean.size, noise_now, rng, Bf, h_harm)
        v_c_rec = R @ (Pif @ (v_f_clean + eta_f))
        rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)
        loss = norm(alpha_r - alpha_t)**2

        # diagnostics
        gap, DK = dk_budget_and_gap(d0c, d1c, M0c, M1c, M2c, d0f, d1f, M0f, M1f, M2f, P, R)

        # conditional projector recalibration
        if (norm(alpha_r - alpha_t) > eps_alpha) or (abs(S_r - S_t) > eps_S) or (DK > dk_max) or (it == 0):
            Pif, Bf = transported_projector(M1f, P, Bc)

            # re-evaluate quickly after recalibration (no extra noise)
            v_c_rec = R @ (Pif @ v_f_clean)
            rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)
            loss = norm(alpha_r - alpha_t)**2

        # finite-difference gradient wrt biases (uses current noise realization)
        def eval_loss(bx, by):
            M1f_tmp = apply_metric_bias(M1f_base, bx, by, Nx_f, Ny_f)
            R_tmp   = restrict_M_adjoint(P, M1c, M1f_tmp)
            Pif_tmp, Bf_tmp = transported_projector(M1f_tmp, P, Bc)
            v_rec_tmp = R_tmp @ (Pif_tmp @ (v_f_clean + eta_f))
            _, _, a_tmp = cycle_weights(M1c, Bc, v_rec_tmp)
            return norm(a_tmp - alpha_t)**2

        eps_fd = 1e-3
        gx = (eval_loss(bias_x + eps_fd, bias_y) - eval_loss(bias_x - eps_fd, bias_y)) / (2*eps_fd)
        gy = (eval_loss(bias_x, bias_y + eps_fd) - eval_loss(bias_x, bias_y - eps_fd)) / (2*eps_fd)
        g  = np.array([gx, gy])

        # Adam update on biases
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g*g)
        mhat = m / (1 - beta1**(it+1))
        vhat = v / (1 - beta2**(it+1))
        step = adam_lr * mhat / (np.sqrt(vhat) + epsAdam)

        bias_x, bias_y = np.clip([bias_x, bias_y] - step, bmin, bmax)

        print(f"{it:02d}\t{loss:.3e}\t{np.round(alpha_r,6)}\t"
              f"{bias_x:.4f}\t{bias_y:.4f}\t{abs(S_r - S_t):.2e}\t{gap:.3e}\t{DK:.3e}")

if __name__ == "__main__":
    # You can tweak these here (this was the missing part that raised NameError)
    run_v9(
        Nx=16, Ny=16,
        alpha_phi=0.9, alpha_n=0.435,
        steps=40,
        noise0=0.03,
        harmonic_noise_frac0=0.20,
        seed=0,
        dk_max=50.0,
        eps_alpha=1e-3,
        eps_S=5e-3,
        bias_bounds=(0.95, 1.05),
        adam_lr=0.02,
    )
