# Transmon_SFIT_Encoder_T2_feedback_v8.py
import numpy as np
from numpy.linalg import norm
from typing import Tuple

# Import fixed operators (must include restrict_M_adjoint)
from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats, laplacian_1form, harmonic_basis,
    cycle_weights, build_prolong_restrict_edges,
    transported_projector, quasiparticle_noise,
    restrict_M_adjoint,
)

# ---------- utilities ----------
def mdot(M, a, b):
    return float(a.T @ (M @ b))

def apply_metric_bias(M1, bias_x=1.0, bias_y=1.0, Nx=16, Ny=16):
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x
    M1b[n:, n:] *= bias_y
    return M1b

def quasiparticle_noise_hybrid(M1f, size, scale, rng, Bf, harmonic_frac=0.0):
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn = np.sqrt(mdot(M1f, h_noise, h_noise)) + 1e-15  # norm of the harmonic component
        h_noise *= (harmonic_frac * scale) / hn
        eta += h_noise
    return eta

# ---------- optimizer ----------
class TransmonOptimizer:
    def __init__(self, Nx=16, Ny=16, alpha_phi=0.9, alpha_n=0.435, seed=0):
        self.Nx, self.Ny = Nx, Ny
        self.Nx_f, self.Ny_f = 2*Nx, 2*Ny
        self.rng = np.random.default_rng(seed)

        # Coarse grid
        self.d0c, self.d1c, self.M0c, self.M1c, self.M2c = build_dec_mats(Nx, Ny)
        self.L1c = laplacian_1form(self.d0c, self.d1c, self.M0c, self.M1c, self.M2c)
        self.Bc, _ = harmonic_basis(self.M1c, self.L1c, k=2)

        # Target state
        self.v_c = alpha_phi * self.Bc[:, 0] + alpha_n * self.Bc[:, 1]
        self.rho_t, self.S_t, self.alpha_t = cycle_weights(self.M1c, self.Bc, self.v_c)

        # Intergrid
        self.P, _R_unused = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")

        # Biases
        self.bias_x = 1.10
        self.bias_y = 0.90

        # Fine base + projector (initialized on first evaluate)
        self._M1f_base = None
        self.Pif = None
        self.Bf = None
        self._proj_current_for_grad = False  # we freeze it inside gradient evals

    # --- internal: ensure fine base matrices available ---
    def _ensure_fine_base(self):
        if self._M1f_base is None:
            d0f, d1f, M0f, M1f_base, M2f = build_dec_mats(self.Nx_f, self.Ny_f)
            self._d0f, self._d1f, self._M0f, self._M2f = d0f, d1f, M0f, M2f
            self._M1f_base = M1f_base

    def _M1f_with_bias(self, bx, by):
        self._ensure_fine_base()
        return apply_metric_bias(self._M1f_base, bx, by, self.Nx_f, self.Ny_f)

    def _update_projector_if_needed(self, M1f, force=False):
        if force or (self.Pif is None) or (self.Bf is None):
            self.Pif, self.Bf = transported_projector(M1f, self.P, self.Bc)

    # --- forward evaluation ---
    def evaluate(self, bx=None, by=None, noise_scale=0.03, harmonic_noise_frac=0.2,
                 eta_override=None, freeze_projector=False):
        """
        Evaluate loss and diagnostics at (bx,by).
        If freeze_projector=True, do not recompute Pif/Bf (used during gradient evaluations).
        eta_override lets us reuse the same noise sample for consistency.
        """
        if bx is None: bx = self.bias_x
        if by is None: by = self.bias_y

        M1f = self._M1f_with_bias(bx, by)
        # M-adjoint restriction for current M1f
        R_curr = restrict_M_adjoint(self.P, self.M1c, M1f)

        # Projector handling
        if not freeze_projector:
            self._update_projector_if_needed(M1f, force=False)

        # Forward channel
        v_f_clean = self.P @ self.v_c

        if eta_override is None:
            eta_f = quasiparticle_noise_hybrid(M1f, v_f_clean.size, noise_scale,
                                               self.rng, self.Bf, harmonic_frac=harmonic_noise_frac)
        else:
            eta_f = eta_override

        v_c_rec = R_curr @ (self.Pif @ (v_f_clean + eta_f))

        rho_r, S_r, alpha_r = cycle_weights(self.M1c, self.Bc, v_c_rec)
        loss = norm(alpha_r - self.alpha_t)**2
        return dict(loss=loss, alpha_rec=alpha_r, Sdr=abs(S_r - self.S_t))

    # --- finite-difference gradient with frozen noise/projector ---
    def fd_grad(self, x, eps=1e-3, noise_scale=0.03, harmonic_noise_frac=0.2):
        """
        Central finite differences for (bias_x, bias_y).
        We FREEZE both noise and projector inside the stencil to keep the landscape smooth.
        """
        bx, by = float(x[0]), float(x[1])

        # freeze projector at the center point
        M1f_center = self._M1f_with_bias(bx, by)
        self._update_projector_if_needed(M1f_center, force=True)

        # freeze a single noise sample at the center
        v_f_clean = self.P @ self.v_c
        eta_frozen = quasiparticle_noise_hybrid(M1f_center, v_f_clean.size, noise_scale,
                                                self.rng, self.Bf, harmonic_frac=harmonic_noise_frac)

        def f(bx_, by_):
            res = self.evaluate(bx_, by_, noise_scale, harmonic_noise_frac,
                                eta_override=eta_frozen, freeze_projector=True)
            return res['loss']

        # central differences
        fxp = f(bx+eps, by); fxm = f(bx-eps, by)
        fyp = f(bx, by+eps); fym = f(bx, by-eps)
        gx = (fxp - fxm) / (2*eps)
        gy = (fyp - fym) / (2*eps)
        return np.array([gx, gy], dtype=float)

    def optimize_step(self, lr=0.1, noise_scale=0.03, harmonic_noise_frac=0.2,
                      clip_lo=0.8, clip_hi=1.2, recalib_loss_factor=1.5, recalib_entropy_tol=1e-6):
        # evaluate at current point
        cur = self.evaluate(self.bias_x, self.bias_y, noise_scale, harmonic_noise_frac)
        prev_loss, prev_Sdr = cur['loss'], cur['Sdr']

        # gradient
        g = self.fd_grad([self.bias_x, self.bias_y], eps=5e-3,
                         noise_scale=noise_scale, harmonic_noise_frac=harmonic_noise_frac)

        # step with clipping
        new_bx = float(np.clip(self.bias_x - lr * g[0], clip_lo, clip_hi))
        new_by = float(np.clip(self.bias_y - lr * g[1], clip_lo, clip_hi))
        self.bias_x, self.bias_y = new_bx, new_by

        # evaluate after update (allow projector update)
        nxt = self.evaluate(self.bias_x, self.bias_y, noise_scale, harmonic_noise_frac)

        # recalibrate projector if things worsened
        if (nxt['loss'] > recalib_loss_factor * prev_loss) or (nxt['Sdr'] > recalib_entropy_tol):
            M1f_now = self._M1f_with_bias(self.bias_x, self.bias_y)
            self._update_projector_if_needed(M1f_now, force=True)  # recalibrate
            nxt = self.evaluate(self.bias_x, self.bias_y, noise_scale, harmonic_noise_frac)

        return nxt

# ---------- main loop ----------
def run_optimization(steps=40, lr=0.15, noise=0.03, harmonic_noise_frac=0.2, seed=0):
    opt = TransmonOptimizer(seed=seed)
    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr")
    for it in range(steps):
        # mild annealing
        current_noise = noise * max(0.4, 0.98**it)
        res = opt.optimize_step(lr=lr, noise_scale=current_noise, harmonic_noise_frac=harmonic_noise_frac)
        print(f"{it:02d}\t{res['loss']:.3e}\t{np.round(res['alpha_rec'],6)}\t{opt.bias_x:.4f}\t{opt.bias_y:.4f}\t{res['Sdr']:.2e}")
        if res['loss'] < 1e-6 and it > 5:
            print("Convergence achieved.")
            break

if __name__ == "__main__":
    run_optimization()
