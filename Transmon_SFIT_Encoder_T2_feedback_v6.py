# Transmon_SFIT_Encoder_T2_feedback_v7.py
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
    restrict_M_adjoint,   # <-- add this
)

# ---------- helpers ----------
def apply_metric_bias(M1, bias_x=1.0, bias_y=1.0, Nx=16, Ny=16):
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x
    M1b[n:, n:] *= bias_y
    return M1b

def mdot(M, a, b): return float(a.T @ (M @ b))

def quasiparticle_noise_hybrid(M1f, size, scale, rng, Bf, harmonic_frac=0.0):
    # orthogonal noise + optional harmonic contamination
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn = np.sqrt(mdot(M1f, h_noise, h_noise)) + 1e-15
        h_noise *= (harmonic_frac * scale) / hn
        eta = eta + h_noise
    return eta

def dk_budget_and_gap(d0c,d1c,M0c,M1c,M2c, d0f,d1f,M0f,M1f,M2f, P,R, k=2):
    Lc = laplacian_1form(d0c,d1c,M0c,M1c,M2c)
    Lf = laplacian_1form(d0f,d1f,M0f,M1f,M2f)
    Mc = cholesky(M1c); Mc_inv = inv(Mc)
    A = Mc_inv.T @ Lc @ Mc_inv
    vals = np.sort(eigh(A, UPLO='U')[0])
    gap = float(vals[k])  # λ_{k+1}
    E = R @ (Lf @ P) - Lc
    Em = Mc_inv.T @ E @ Mc_inv
    DK = float(norm(Em, 'fro')) / max(gap, 1e-15)   # proxy for ||·||_{2,M}/gap
    return gap, DK

def eval_loss_and_state(bx, by, *,
                        P,R,Bc,M1c, v_c, v_c_clean, alpha_t, rho_t, S_t,
                        eta_seed, noise, harmonic_noise_frac,
                        Nx_f, Ny_f, recalibrate=False):
    """
    One evaluation at (bx,by). If recalibrate=True, recompute Pif,Bf at current M1f,
    otherwise use transported projector fresh each call (consistent).
    """
    d0f,d1f,M0f,M1f,M2f = build_dec_mats(Nx_f, Ny_f)
    M1f = apply_metric_bias(M1f, bias_x=bx, bias_y=by, Nx=Nx_f, Ny=Ny_f)
    Pif, Bf = transported_projector(M1f, P, Bc)  # transported harmonic projector
    rng = np.random.default_rng(eta_seed)
    eta_f = quasiparticle_noise_hybrid(M1f, v_c_clean.size, noise, rng, Bf, harmonic_noise_frac)
    v_c_rec = R @ (Pif @ (v_c_clean + eta_f))

    rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)

    # Diagnostics
    err_alpha = norm(alpha_r - alpha_t)
    err_M = np.sqrt( mdot(M1c, v_c_rec - v_c, v_c_rec - v_c) )

    return (v_c_rec, alpha_r, rho_r, S_r, err_alpha, err_M, M1f, Pif, Bf)

# ---------- Adam optimizer on (bias_x, bias_y) ----------
class Adam2D:
    def __init__(self, lr=0.25, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(2); self.v = np.zeros(2); self.t = 0
    def step(self, x, g):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*g
        self.v = self.b2*self.v + (1-self.b2)*(g*g)
        mhat = self.m/(1-self.b1**self.t)
        vhat = self.v/(1-self.b2**self.t)
        x_new = x - self.lr * mhat / (np.sqrt(vhat)+self.eps)
        return x_new

# ---------- main ----------
def run_feedback(
    Nx=16, Ny=16,
    alpha_phi=0.9, alpha_n=0.435,
    steps=40, seed=0,
    # annealing schedule: (noise, harmonic_frac, T)
    schedule=((0.005, 0.0, 10), (0.02, 0.1, 15), (0.03, 0.2, 15)),
    fd_h=2.5e-3,         # finite-diff step for gradients
    lr=0.25,             # Adam LR
    # NEW: loss weights
    w_alpha=10.0, w_rho=0.1, w_S=1.0, w_bias=0.05,
    # NEW: controls
    recal_thresh=1e-6,   # trigger projector recalibration if err exceeds this
    bias_box=(0.8, 1.2)  # tighter constraints on biases
):
    rng = np.random.default_rng(seed)

    # coarse reference / targets
    d0c,d1c,M0c,M1c,M2c = build_dec_mats(Nx, Ny)
    L1c = laplacian_1form(d0c,d1c,M0c,M1c,M2c)
    Bc,_ = harmonic_basis(M1c, L1c, k=2)
    v_c = alpha_phi*Bc[:,0] + alpha_n*Bc[:,1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # intergrid maps
    P,R = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")
    Nx_f, Ny_f = 2*Nx, 2*Ny
    v_c_clean = P @ v_c

    # optimizer
    adam = Adam2D(lr=lr)
    x = np.array([1.10, 0.90])  # biases

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr\t\tgap\t\tDK_budget\tRECAL")
    it = 0
    for noise, hfrac, T in schedule:
        for _ in range(T):
            eta_seed = 1000 + it

            # evaluate state
            v_c_rec, alpha_r, rho_r, S_r, err_alpha, err_M, M1f, Pif, Bf = eval_loss_and_state(
                x[0], x[1],
                P=P, R=R, Bc=Bc, M1c=M1c,
                v_c=v_c, v_c_clean=v_c_clean,
                alpha_t=alpha_t, rho_t=rho_t, S_t=S_t,
                eta_seed=eta_seed, noise=noise, harmonic_noise_frac=hfrac,
                Nx_f=Nx_f, Ny_f=Ny_f
            )

            # diagnostics: gap & DK
            d0f,d1f,M0f,_,M2f = build_dec_mats(Nx_f, Ny_f)
            gap, DK = dk_budget_and_gap(d0c,d1c,M0c,M1c,M2c, d0f,d1f,M0f,M1f,M2f, P,R)

            # loss (with entropy + bias penalties)
            loss = (
                w_alpha * norm(alpha_r - alpha_t)**2 +
                w_rho   * norm(rho_r   - rho_t  )**2 +
                w_S     * (S_r - S_t)**2 +
                w_bias  * ((x[0]-1.0)**2 + (x[1]-1.0)**2)
            )

            # check if we should recalibrate projector (symbolic “re-anchoring”)
            recal = False
            if (err_M > recal_thresh) or (err_alpha > recal_thresh):
                # recompute projector with CURRENT metric and suppress harmonic noise once
                recal = True
                v_c_rec, alpha_r, rho_r, S_r, err_alpha, err_M, M1f, Pif, Bf = eval_loss_and_state(
                    x[0], x[1],
                    P=P, R=R, Bc=Bc, M1c=M1c,
                    v_c=v_c, v_c_clean=v_c_clean,
                    alpha_t=alpha_t, rho_t=rho_t, S_t=S_t,
                    eta_seed=eta_seed, noise=noise, harmonic_noise_frac=0.0,  # one-step pure re-anchor
                    Nx_f=Nx_f, Ny_f=Ny_f, recalibrate=True
                )
                # refresh loss with updated state
                loss = (
                    w_alpha * norm(alpha_r - alpha_t)**2 +
                    w_rho   * norm(rho_r   - rho_t  )**2 +
                    w_S     * (S_r - S_t)**2 +
                    w_bias  * ((x[0]-1.0)**2 + (x[1]-1.0)**2)
                )

            # finite-diff gradient on biases (same eta_seed for fair gradient)
            def loss_at(bx, by):
                v_c_rec2, a2, r2, S2, ea2, em2, *_ = eval_loss_and_state(
                    bx, by,
                    P=P, R=R, Bc=Bc, M1c=M1c,
                    v_c=v_c, v_c_clean=v_c_clean,
                    alpha_t=alpha_t, rho_t=rho_t, S_t=S_t,
                    eta_seed=eta_seed, noise=noise, harmonic_noise_frac=hfrac,
                    Nx_f=Nx_f, Ny_f=Ny_f
                )
                return (
                    w_alpha * norm(a2 - alpha_t)**2 +
                    w_rho   * norm(r2 - rho_t  )**2 +
                    w_S     * (S2 - S_t)**2 +
                    w_bias  * ((bx-1.0)**2 + (by-1.0)**2)
                )

            h = 2.5e-3
            fxp = loss_at(x[0]+h, x[1]); fxm = loss_at(x[0]-h, x[1])
            fyp = loss_at(x[0], x[1]+h); fym = loss_at(x[0], x[1]-h)
            gx = (fxp - fxm)/(2*h); gy = (fyp - fym)/(2*h)

            # Adam step + tight box
            x = adam.step(x, np.array([gx, gy]))
            x = np.clip(x, bias_box[0], bias_box[1])

            Sdr = abs(S_r - S_t)
            print(f"{it:02d}\t{loss:.3e}\t{np.round(alpha_r,6)}\t{x[0]:.4f}\t{x[1]:.4f}\t"
                  f"{Sdr:.2e}\t{gap:.3e}\t{DK:.3e}\t{('RECAL' if recal else '')}")

            it += 1
            if it >= steps: break
        if it >= steps: break

    # (optional) simple telemetry summary could go here
if __name__ == '__main__':
    run_feedback()
