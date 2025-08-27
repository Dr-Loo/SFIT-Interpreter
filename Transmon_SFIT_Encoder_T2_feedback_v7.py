# Transmon_SFIT_Encoder_T2_feedback_v7.py
import numpy as np
from numpy.linalg import eigh, norm, cholesky, inv
from scipy.optimize import minimize
from typing import Tuple

# Import fixed operators
from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats, laplacian_1form, harmonic_basis,
    cycle_weights, build_prolong_restrict_edges,
    transported_projector, quasiparticle_noise,
    restrict_M_adjoint,                 # <-- NEW: metric-aware restriction
)

# ---------------- utilities ----------------
def mdot(M, a, b):
    """M-weighted inner product a^T M b."""
    return float(a.T @ (M @ b))

def complex_step_gradient(f, x, h=1e-20):
    """Machine-precision gradient via complex-step."""
    grad = np.zeros_like(x, dtype=np.complex128)
    for i in range(len(x)):
        x_perturbed = x.copy()
        x_perturbed[i] += 1j * h
        grad[i] = np.imag(f(x_perturbed)) / h
    return np.real(grad)

def apply_metric_bias(M1, bias_x=1.0, bias_y=1.0, Nx=16, Ny=16):
    """Apply anisotropic scaling to edge masses."""
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x  # x-edges
    M1b[n:, n:] *= bias_y  # y-edges
    return M1b

def quasiparticle_noise_hybrid(M1f, size, scale, rng, Bf, harmonic_frac=0.0):
    """Noise with quasiparticle + optional harmonic components."""
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn_norm = np.sqrt(mdot(M1f, h_noise, h_noise)) + 1e-15  # <-- FIX: norm of h_noise, not eta
        h_noise *= (harmonic_frac * scale) / hn_norm
        eta += h_noise
    return eta

# ---------------- optimizer ----------------
class TransmonOptimizer:
    def __init__(self, Nx=16, Ny=16, alpha_phi=0.9, alpha_n=0.435):
        # Coarse grid setup
        self.Nx, self.Ny = Nx, Ny
        self.d0c, self.d1c, self.M0c, self.M1c, self.M2c = build_dec_mats(Nx, Ny)
        self.L1c = laplacian_1form(self.d0c, self.d1c, self.M0c, self.M1c, self.M2c)
        self.Bc, _ = harmonic_basis(self.M1c, self.L1c, k=2)

        # Target state
        self.v_c = alpha_phi * self.Bc[:, 0] + alpha_n * self.Bc[:, 1]
        self.rho_t, self.S_t, self.alpha_t = cycle_weights(self.M1c, self.Bc, self.v_c)

        # Fine grid base (unbiased)
        self.Nx_f, self.Ny_f = 2*Nx, 2*Ny
        self.P, _R_unused = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")  # base P reused

        # Bias state + projector control
        self.bias_x = 1.10
        self.bias_y = 0.90
        self.projector_recalibrated = False
        self.Pif = None
        self.Bf = None

    def apply_biases(self, M1f_base):
        """Apply current biases to fine grid metric."""
        return apply_metric_bias(M1f_base, self.bias_x, self.bias_y, self.Nx_f, self.Ny_f)

    def update_projector(self, M1f):
        """Recompute the fine projector (transported) when flagged."""
        if (not self.projector_recalibrated) or (self.Pif is None):
            self.Pif, self.Bf = transported_projector(M1f, self.P, self.Bc)
            self.projector_recalibrated = True

    def evaluate(self, noise_scale=0.03, harmonic_noise_frac=0.2, rng=None):
        """One forward pass at current biases; returns diagnostics."""
        if rng is None:
            rng = np.random.default_rng()

        # Build fine operators and metric for CURRENT biases
        d0f, d1f, M0f, M1f_base, M2f = build_dec_mats(self.Nx_f, self.Ny_f)
        M1f = self.apply_biases(M1f_base)

        # (1) Recompute metric-adjoint restriction for CURRENT M1f  <-- KEY FIX
        R_curr = restrict_M_adjoint(self.P, self.M1c, M1f)

        # (2) Update (or keep) projector as needed                   <-- KEY FIX
        self.update_projector(M1f)

        # Forward channel + noise
        v_f_clean = self.P @ self.v_c
        eta_f = quasiparticle_noise_hybrid(M1f, v_f_clean.size, noise_scale, rng, self.Bf, harmonic_frac=harmonic_noise_frac)
        v_c_rec = R_curr @ (self.Pif @ (v_f_clean + eta_f))

        # Diagnostics
        rho_r, S_r, alpha_r = cycle_weights(self.M1c, self.Bc, v_c_rec)
        loss = norm(alpha_r - self.alpha_t)**2

        return {
            'loss': loss,
            'alpha_rec': alpha_r,
            'Sdr': abs(S_r - self.S_t),
            'M1f': M1f,
            'v_rec': v_c_rec,
            'R_curr': R_curr,
        }

    def optimize_step(self, lr=0.1, noise_scale=0.03, max_bias_change=0.02,
                      recalib_loss_factor=1.5, recalib_entropy_tol=1e-6):
        """One optimization step with gradient, clipping, and auto-recalibration."""
        prev = self.evaluate(noise_scale)
        prev_loss, prev_Sdr = prev['loss'], prev['Sdr']

        # Complex-step gradient wrt (bias_x, bias_y)
        def f(bx_by):
            bx, by = float(bx_by[0]), float(bx_by[1])
            old_bx, old_by = self.bias_x, self.bias_y
            # temporary set
            self.bias_x, self.bias_y = bx, by
            val = self.evaluate(noise_scale)['loss']
            # restore
            self.bias_x, self.bias_y = old_bx, old_by
            return val

        grad = complex_step_gradient(f, np.array([self.bias_x, self.bias_y]))

        # Update with clipping (physical bounds)                    <-- KEY FIX
        new_bx = float(np.clip(self.bias_x - lr * grad[0], 0.8, 1.2))
        new_by = float(np.clip(self.bias_y - lr * grad[1], 0.8, 1.2))
        self.bias_x, self.bias_y = new_bx, new_by

        # Evaluate after update
        cur = self.evaluate(noise_scale)
        # Trigger projector recalibration on loss or entropy jump   <-- KEY FIX
        if (cur['loss'] > recalib_loss_factor * prev_loss) or (cur['Sdr'] > recalib_entropy_tol):
            self.projector_recalibrated = False
            cur = self.evaluate(noise_scale)

        return cur

# ---------------- main loop ----------------
def run_optimization(steps=40, lr=0.1, noise=0.03, harmonic_noise_frac=0.2, seed=0):
    opt = TransmonOptimizer()
    rng = np.random.default_rng(seed)

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr")
    for it in range(steps):
        # Gentle noise annealing (optional)
        current_noise = noise * max(0.3, 0.98**it)

        result = opt.optimize_step(
            lr=lr,
            noise_scale=current_noise,
            max_bias_change=0.01,
            recalib_loss_factor=1.5,
            recalib_entropy_tol=1e-6,
        )

        print(f"{it:02d}\t{result['loss']:.3e}\t{np.round(result['alpha_rec'],6)}\t"
              f"{opt.bias_x:.4f}\t{opt.bias_y:.4f}\t{result['Sdr']:.2e}")

        if result['loss'] < 1e-8 and it > 5:
            print("Convergence achieved.")
            break

if __name__ == "__main__":
    run_optimization()
