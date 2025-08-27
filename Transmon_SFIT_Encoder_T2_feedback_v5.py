# Transmon_SFIT_Encoder_T2_feedback_v5.py
# Fixes vs v4:
#  - Use FIXED targets (alpha_t, rho_t) from coarse grid in the loss
#  - Rebuild transported projector for each (bias_x, bias_y)
#  - Two-sided finite differences + simple backtracking line search

import numpy as np
from numpy.linalg import eigh, norm, cholesky, inv
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

# ---------- helpers ----------
def apply_metric_bias(M1: np.ndarray, bias_x: float = 1.0, bias_y: float = 1.0,
                     Nx: int = 16, Ny: int = 16) -> np.ndarray:
    """Scale x- and y-edge blocks of M1 (anisotropy / detuning)."""
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x  # x-edges
    M1b[n:, n:] *= bias_y  # y-edges
    return M1b

def mdot(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(a.T @ (M @ b))

def quasiparticle_noise_hybrid(M1f: np.ndarray, size: int, scale: float,
                               rng: np.random.Generator, Bf: np.ndarray,
                               harmonic_frac: float = 0.0) -> np.ndarray:
    """Orthogonal noise + optional H^1 contamination."""
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn = np.sqrt(mdot(M1f, h_noise, h_noise)) + 1e-15
        h_noise *= (harmonic_frac * scale) / hn
        eta = eta + h_noise
    return eta

def dk_budget_and_gap(d0c: np.ndarray, d1c: np.ndarray, M0c: np.ndarray, M1c: np.ndarray, M2c: np.ndarray,
                      d0f: np.ndarray, d1f: np.ndarray, M0f: np.ndarray, M1f: np.ndarray, M2f: np.ndarray,
                      P: np.ndarray, R: np.ndarray, k: int = 2) -> Tuple[float, float]:
    """Spectral gap of Δ1 (coarse) and Davis–Kahan budget ‖R Δf P − Δc‖_2,M / gap."""
    Lc = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Lf = laplacian_1form(d0f, d1f, M0f, M1f, M2f)

    Mc_half = cholesky(M1c)
    Mc_inv = inv(Mc_half)
    A = Mc_inv.T @ Lc @ Mc_inv
    vals = np.sort(eigh(A, UPLO='U')[0])
    gap = float(vals[k])  # λ_{k+1}

    E = R @ (Lf @ P) - Lc
    Em = Mc_inv.T @ E @ Mc_inv
    dk = float(norm(Em, 'fro')) / max(gap, 1e-15)  # Frobenius proxy
    return gap, dk

# ---------- objective & gradient ----------
def eval_loss(bx: float, by: float, *,
              P, R, Bc, M1c,
              v_c_clean: np.ndarray,
              alpha_t: np.ndarray, rho_t: np.ndarray,
              eta_seed: int,
              noise: float,
              harmonic_noise_frac: float,
              Nx_f: int, Ny_f: int) -> Tuple[float, np.ndarray, float]:
    """
    Build fine metric for (bx,by), rebuild projector, push forward + noise,
    pull back, and compute loss to FIXED targets (alpha_t, rho_t).
    Returns: loss, recovered alpha_r, entropy S_r (for logging).
    """
    # Fine DEC mats
    d0f, d1f, M0f, M1f, M2f = build_dec_mats(Nx_f, Ny_f)
    M1f = apply_metric_bias(M1f, bias_x=bx, bias_y=by, Nx=Nx_f, Ny=Ny_f)

    # Transported projector must be rebuilt to depend on (bx, by)
    Pif, Bf = transported_projector(M1f, P, Bc)

    # Common noise across calls for fair finite differences
    rng = np.random.default_rng(eta_seed)
    eta_f = quasiparticle_noise_hybrid(M1f, v_c_clean.size, noise, rng, Bf, harmonic_noise_frac)

    v_c_rec = R @ (Pif @ (v_c_clean + eta_f))
    rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)

    # LOSS vs FIXED targets from coarse grid
    loss = 10.0 * norm(alpha_r - alpha_t)**2 + 0.1 * norm(rho_r - rho_t)**2
    return float(loss), alpha_r, float(S_r)

def finite_diff_grad(bx: float, by: float, h: float, **kwargs) -> Tuple[float, float]:
    f_xp, _, _ = eval_loss(bx + h, by, **kwargs)
    f_xm, _, _ = eval_loss(bx - h, by, **kwargs)
    f_yp, _, _ = eval_loss(bx, by + h, **kwargs)
    f_ym, _, _ = eval_loss(bx, by - h, **kwargs)
    gx = (f_xp - f_xm) / (2*h)
    gy = (f_yp - f_ym) / (2*h)
    return gx, gy

# ---------- main loop ----------
def run_feedback(
    Nx: int = 16,
    Ny: int = 16,
    alpha_phi: float = 0.9,
    alpha_n: float = 0.435,
    noise: float = 0.03,
    harmonic_noise_frac: float = 0.2,
    steps: int = 20,
    seed: int = 0,
    fd_h: float = 1e-3,
    lr: float = 0.5,
    backtrack: int = 6,   # backtracking steps if loss doesn't drop
):
    rng = np.random.default_rng(seed)

    # Coarse reference
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    L1c = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Bc, _ = harmonic_basis(M1c, L1c, k=2)
    v_c = alpha_phi * Bc[:, 0] + alpha_n * Bc[:, 1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # Intergrid maps (M-isometry recommended)
    P, R = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")

    # Fine grid init
    Nx_f, Ny_f = 2 * Nx, 2 * Ny
    bias_x, bias_y = 1.10, 0.90
    d0f, d1f, M0f, M1f, M2f = build_dec_mats(Nx_f, Ny_f)
    M1f = apply_metric_bias(M1f, bias_x=bias_x, bias_y=bias_y, Nx=Nx_f, Ny=Ny_f)
    Pif, Bf = transported_projector(M1f, P, Bc)

    v_f_clean = P @ v_c

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr\t\tgap\t\tDK_budget")
    for it in range(steps):
        eta_seed = 1000 + it  # common noise seed for fair FD

        # Current gap & DK (diagnostic)
        d0f_cur, d1f_cur, M0f_cur, M1f_cur, M2f_cur = build_dec_mats(Nx_f, Ny_f)
        M1f_cur = apply_metric_bias(M1f_cur, bias_x=bias_x, bias_y=bias_y, Nx=Nx_f, Ny=Ny_f)
        gap, DK = dk_budget_and_gap(d0c, d1c, M0c, M1c, M2c,
                                    d0f_cur, d1f_cur, M0f_cur, M1f_cur, M2f_cur, P, R)

        # Current loss & state
        loss, alpha_r, S_r = eval_loss(
            bias_x, bias_y,
            P=P, R=R, Bc=Bc, M1c=M1c,
            v_c_clean=v_f_clean,
            alpha_t=alpha_t, rho_t=rho_t,
            eta_seed=eta_seed,
            noise=noise,
            harmonic_noise_frac=harmonic_noise_frac,
            Nx_f=Nx_f, Ny_f=Ny_f
        )

        # Gradient (two-sided FD)
        gx, gy = finite_diff_grad(
            bias_x, bias_y, fd_h,
            P=P, R=R, Bc=Bc, M1c=M1c,
            v_c_clean=v_f_clean,
            alpha_t=alpha_t, rho_t=rho_t,
            eta_seed=eta_seed,
            noise=noise,
            harmonic_noise_frac=harmonic_noise_frac,
            Nx_f=Nx_f, Ny_f=Ny_f
        )

        # Backtracking line search on (bias_x, bias_y)
        step = lr
        for _ in range(backtrack):
            bx_try = bias_x - step * gx
            by_try = bias_y - step * gy
            # (optional) clamp biases a bit to avoid silly scales
            bx_try = float(np.clip(bx_try, 0.7, 1.4))
            by_try = float(np.clip(by_try, 0.7, 1.4))

            loss_try, _, _ = eval_loss(
                bx_try, by_try,
                P=P, R=R, Bc=Bc, M1c=M1c,
                v_c_clean=v_f_clean,
                alpha_t=alpha_t, rho_t=rho_t,
                eta_seed=eta_seed,
                noise=noise,
                harmonic_noise_frac=harmonic_noise_frac,
                Nx_f=Nx_f, Ny_f=Ny_f
            )
            if loss_try < loss:
                bias_x, bias_y = bx_try, by_try
                loss = loss_try
                break
            step *= 0.5  # shrink step

        Sdr = abs(S_r - S_t)
        print(f"{it:02d}\t{loss:.3e}\t{np.round(alpha_r,6)}\t{bias_x:.4f}\t{bias_y:.4f}\t{Sdr:.2e}\t{gap:.3e}\t{DK:.3e}")

if __name__ == "__main__":
    run_feedback(
        Nx=16, Ny=16,
        alpha_phi=0.9, alpha_n=0.435,
        noise=0.03, harmonic_noise_frac=0.2,
        steps=20, seed=0,
        fd_h=1e-3, lr=0.5, backtrack=6,
    )
