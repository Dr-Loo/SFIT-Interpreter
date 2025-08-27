# Transmon_SFIT_Encoder_T2_feedback.py
import numpy as np

from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats, laplacian_1form, harmonic_basis, project_harmonic,
    cycle_weights, build_prolong_restrict_edges, transported_projector,
    quasiparticle_noise
)

def apply_metric_bias(M1, bias_x=1.0, bias_y=1.0, Nx=16, Ny=16):
    """Scale x-edges and y-edges blocks of M1 (simulates anisotropy / detuning)."""
    M1b = M1.copy()
    n = Nx*Ny
    M1b[:n, :n] *= bias_x
    M1b[n:, n:] *= bias_y
    return M1b

def run_feedback(Nx=16, Ny=16, alpha_phi=0.9, alpha_n=0.435,
                 noise=0.03, seed=0, steps=20, lr=0.5):
    rng = np.random.default_rng(seed)

    # coarse (fixed)
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    L1c = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Bc, _ = harmonic_basis(M1c, L1c, k=2)
    Pic = Bc @ (Bc.T @ M1c)
    v_c = alpha_phi * Bc[:,0] + alpha_n * Bc[:,1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # start with a biased fine metric
    bias_x, bias_y = 1.10, 0.90  # intentionally skewed
    for it in range(steps):
        # fine DEC under current bias
        d0f, d1f, M0f, M1f, M2f = build_dec_mats(2*Nx, 2*Ny)
        M1f = apply_metric_bias(M1f, bias_x=bias_x, bias_y=bias_y, Nx=2*Nx, Ny=2*Ny)
        L1f = laplacian_1form(d0f, d1f, M0f, M1f, M2f)

        # intergrid maps
        from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import build_prolong_restrict_edges
        P, R = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")

        # transported projector using current M1f
        from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import transported_projector
        Pif, Bf = transported_projector(M1f, P, Bc)

        v_f_clean = P @ v_c
        eta_f = quasiparticle_noise(M1f, v_f_clean.size, scale=noise, rng=rng, B=Bf)
        v_c_rec = R @ (Pif @ (v_f_clean + eta_f))

        rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)
        loss = np.linalg.norm(alpha_r - alpha_t)**2

        # simple finite-diff "gradient" on biases
        def eval_loss(bx, by):
            M1f2 = apply_metric_bias(M1f, bias_x=bx, bias_y=by, Nx=2*Nx, Ny=2*Ny)
            Pif2, Bf2 = transported_projector(M1f2, P, Bc)
            v_c_rec2 = R @ (Pif2 @ (v_f_clean + eta_f))
            _, _, a2 = cycle_weights(M1c, Bc, v_c_rec2)
            return np.linalg.norm(a2 - alpha_t)**2

        eps = 1e-2
        gx = (eval_loss(bias_x+eps, bias_y) - eval_loss(bias_x-eps, bias_y)) / (2*eps)
        gy = (eval_loss(bias_x, bias_y+eps) - eval_loss(bias_x, bias_y-eps)) / (2*eps)

        bias_x -= lr * gx
        bias_y -= lr * gy

        print(f"it={it:02d}  loss={loss:.3e}  alpha_rec={alpha_r}  bias=({bias_x:.4f},{bias_y:.4f})  Sdr={abs(S_r-S_t):.2e}")

if __name__ == "__main__":
    run_feedback()
