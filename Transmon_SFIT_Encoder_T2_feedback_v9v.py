# Transmon_SFIT_Encoder_T2_feedback_v9b.py
# v9b = v9 + (a) dual alpha/rho loss, (b) L2 bias penalty,
#        (c) periodic projector refresh, (d) wider-but-regularized bounds.

import numpy as np
from numpy.linalg import cholesky, inv, norm, eigh

# Import fixed operators from your module with M-isometry P/R and projector transport
from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats,
    laplacian_1form,
    harmonic_basis,
    cycle_weights,
    build_prolong_restrict_edges,
    transported_projector,
    quasiparticle_noise,
)

# --- v9b: training / regularization knobs ---
W_ALPHA = 1.0          # weight for alpha loss
W_RHO   = 0.25         # weight for rho (cycle weight) loss
L2_BIAS = 5e-3         # L2 penalty on (bias_x-1)^2+(bias_y-1)^2
REFRESH_EVERY = 5      # recompute fine projector every N iters
BIAS_MIN, BIAS_MAX = 0.85, 1.15  # wider bounds, but regularized
FD_EPS = 1e-3          # finite-diff step for bias gradients
LR = 0.25              # learning rate for biases

def apply_metric_bias(M1, bias_x=1.0, bias_y=1.0, Nx=16, Ny=16):
    """Anisotropic scaling of the 1-form mass matrix blocks (x-edges then y-edges)."""
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= float(bias_x)
    M1b[n:, n:] *= float(bias_y)
    return M1b

def mdot(M, a, b):
    return float(a.T @ (M @ b))

def quasiparticle_noise_hybrid(M1f, size, scale, rng, Bf, harmonic_frac=0.0):
    """
    Noise mostly orthogonal to H^1 plus optional harmonic contamination
    to force the controller to act.
    """
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn = np.sqrt(max(mdot(M1f, h_noise, h_noise), 1e-18))
        h_noise *= (harmonic_frac * scale) / hn
        eta = eta + h_noise
    return eta

def dk_budget_and_gap(Lc, M1c, d0f, d1f, M0f, M1f, M2f, P, R, k=2):
    """
    gap = λ_{k+1}(Lc x = λ M1c x)
    DK ≈ || M_c^{-1/2} (R Δf P − Lc) M_c^{-1/2} ||_F / gap
    """
    # generalized eigen gap on coarse
    Mc_half = cholesky(M1c)
    Mc_half_inv = inv(Mc_half)
    A = Mc_half_inv.T @ Lc @ Mc_half_inv
    vals = np.sort(eigh(A, UPLO='U')[0])
    gap = float(vals[k])  # after k zero modes on T^2

    # fine Δ with current M1f
    Lf = laplacian_1form(d0f, d1f, M0f, M1f, M2f)
    E = R @ (Lf @ P) - Lc
    Em = Mc_half_inv.T @ E @ Mc_half_inv
    fro = float(norm(Em, 'fro'))
    DK = fro / max(gap, 1e-15)
    return gap, DK

def run_v9b(
    Nx=16,
    Ny=16,
    steps=40,
    noise=0.03,
    harmonic_noise_frac=0.20,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # --- Coarse grid & target symbolic state ---
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    L1c = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Bc, _ = harmonic_basis(M1c, L1c, k=2)

    alpha_phi, alpha_n = 0.9, 0.435
    v_c = alpha_phi * Bc[:, 0] + alpha_n * Bc[:, 1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # --- Inter-grid operators (M-isometry ensures R P = I in M1) ---
    P, R = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")

    # --- Fine grid (base operators) ---
    Nx_f, Ny_f = 2 * Nx, 2 * Ny
    d0f, d1f, M0f, M1f_base, M2f = build_dec_mats(Nx_f, Ny_f)

    # initial biases and metric
    bias_x, bias_y = 1.10, 0.90
    M1f = apply_metric_bias(M1f_base, bias_x, bias_y, Nx_f, Ny_f)

    # initial transported projector on fine
    Pif, Bf = transported_projector(M1f, P, Bc)

    # precompute clean forward (no need to recompute each iter)
    v_f_clean = P @ v_c

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr\t\tgap\t\tDK_budget")
    for it in range(steps):
        # --- noise annealing to 0 by ~80% of schedule ---
        noise_now = noise * max(0.0, 1.0 - it / (0.8 * steps))

        # periodic projector refresh to keep metric/projection aligned
        if it % REFRESH_EVERY == 0:
            Pif, Bf = transported_projector(M1f, P, Bc)

        # forward pass with current metric & projector
        eta_f = quasiparticle_noise_hybrid(
            M1f, v_f_clean.size, scale=noise_now, rng=rng, Bf=Bf, harmonic_frac=harmonic_noise_frac
        )
        v_c_rec = R @ (Pif @ (v_f_clean + eta_f))
        rho_rec, S_r, alpha_rec = cycle_weights(M1c, Bc, v_c_rec)

        # dual objective + L2 bias regularization
        loss_alpha = norm(alpha_rec - alpha_t) ** 2
        loss_rho   = norm(rho_rec   - rho_t  ) ** 2
        loss_l2    = (bias_x - 1.0) ** 2 + (bias_y - 1.0) ** 2
        loss = W_ALPHA * loss_alpha + W_RHO * loss_rho + L2_BIAS * loss_l2

        # finite-difference gradients on biases (reuse Pif this iter)
        def eval_loss(bx, by):
            M1f_tmp = apply_metric_bias(M1f_base, bx, by, Nx_f, Ny_f)
            # reuse Pif/Bf inside this iter; periodic refresh will realign soon
            v_c_rec_tmp = R @ (Pif @ (v_f_clean + eta_f))
            rho_tmp, _, alpha_tmp = cycle_weights(M1c, Bc, v_c_rec_tmp)
            la = norm(alpha_tmp - alpha_t) ** 2
            lr = norm(rho_tmp   - rho_t  ) ** 2
            l2 = (bx - 1.0) ** 2 + (by - 1.0) ** 2
            return W_ALPHA * la + W_RHO * lr + L2_BIAS * l2

        gx = (eval_loss(bias_x + FD_EPS, bias_y) - eval_loss(bias_x - FD_EPS, bias_y)) / (2 * FD_EPS)
        gy = (eval_loss(bias_x, bias_y + FD_EPS) - eval_loss(bias_x, bias_y - FD_EPS)) / (2 * FD_EPS)

        # bias update with wider but clipped bounds
        bias_x = float(np.clip(bias_x - LR * gx, BIAS_MIN, BIAS_MAX))
        bias_y = float(np.clip(bias_y - LR * gy, BIAS_MIN, BIAS_MAX))

        # update metric for next iteration (projector may refresh next loop)
        M1f = apply_metric_bias(M1f_base, bias_x, bias_y, Nx_f, Ny_f)

        # diagnostics: gap & DK budget (cheap on small grids)
        gap, DK = dk_budget_and_gap(L1c, M1c, d0f, d1f, M0f, M1f, M2f, P, R, k=2)

        Sdr = abs(S_r - S_t)
        print(f"{it:02d}\t{loss:.3e}\t{np.round(alpha_rec,6)}\t"
              f"{bias_x:.4f}\t{bias_y:.4f}\t{Sdr:.2e}\t{gap:.3e}\t{DK:.3e}")

if __name__ == "__main__":
    run_v9b(
        Nx=16,
        Ny=16,
        steps=40,
        noise=0.03,
        harmonic_noise_frac=0.20,
        seed=0,
    )
