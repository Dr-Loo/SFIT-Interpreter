# Transmon_SFIT_Encoder_T2_feedback_v3.py
import numpy as np
from numpy.linalg import eigh, norm, cholesky, inv
from scipy.optimize import minimize
from typing import Tuple

# Import your fixed underscore-named module
from Transmon_SFIT_Encoder_T2_fix_M_Adjoint import (
    build_dec_mats,
    laplacian_1form,
    harmonic_basis,
    cycle_weights,
    build_prolong_restrict_edges,
    transported_projector,
    quasiparticle_noise,
)

def apply_metric_bias(M1: np.ndarray, bias_x: float = 1.0, bias_y: float = 1.0, 
                     Nx: int = 16, Ny: int = 16) -> np.ndarray:
    """Apply anisotropic bias to x- and y-edge blocks of M1 mass matrix."""
    M1b = M1.copy()
    n = Nx * Ny
    M1b[:n, :n] *= bias_x  # x-edges
    M1b[n:, n:] *= bias_y  # y-edges
    return M1b

def mdot(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Compute M-weighted inner product a^T M b."""
    return float(a.T @ (M @ b))

def quasiparticle_noise_hybrid(M1f: np.ndarray, size: int, scale: float, rng: np.random.Generator,
                              Bf: np.ndarray, harmonic_frac: float = 0.0) -> np.ndarray:
    """Generate noise with both quasiparticle and harmonic components."""
    eta = quasiparticle_noise(M1f, size, scale=scale, rng=rng, B=Bf)
    if harmonic_frac > 0:
        coeffs = rng.normal(size=Bf.shape[1])
        h_noise = Bf @ coeffs
        hn_norm = np.sqrt(mdot(M1f, h_noise, h_noise)) + 1e-15
        h_noise *= (harmonic_frac * scale) / hn_norm
        eta = eta + h_noise
    return eta

def dk_budget_and_gap(d0c: np.ndarray, d1c: np.ndarray, M0c: np.ndarray, M1c: np.ndarray, M2c: np.ndarray,
                      d0f: np.ndarray, d1f: np.ndarray, M0f: np.ndarray, M1f: np.ndarray, M2f: np.ndarray,
                      P: np.ndarray, R: np.ndarray, k: int = 2) -> Tuple[float, float]:
    """Compute spectral gap and Davis-Kahan budget for operator consistency."""
    Lc = laplacian_1form(d0c, d1c, M0c, M1c, M2c)
    Lf = laplacian_1form(d0f, d1f, M0f, M1f, M2f)

    # Generalized eigenvalues via M1c^{-1/2} similarity transform
    Mc_half = cholesky(M1c)
    Mc_half_inv = inv(Mc_half)
    A = Mc_half_inv.T @ Lc @ Mc_half_inv
    vals = np.sort(eigh(A, UPLO='U')[0])
    gap = float(vals[k])  # λ_{k+1} after k zero modes

    # M-weighted norm of commutator defect E = RΔfP - Δc
    E = R @ (Lf @ P) - Lc
    Em = Mc_half_inv.T @ E @ Mc_half_inv
    fro_norm = float(norm(Em, 'fro'))  # Proxy for ||·||_{2,M}
    DK = fro_norm / max(gap, 1e-15)
    return gap, DK

def complex_step_gradient(f, x: np.ndarray, h: float = 1e-20) -> np.ndarray:
    """Compute machine-precision gradient using complex-step differentiation."""
    grad = np.zeros_like(x, dtype=np.complex128)
    for i in range(len(x)):
        x_perturbed = x.copy()
        x_perturbed[i] += 1j * h
        grad[i] = np.imag(f(x_perturbed)) / h
    return np.real(grad)

class OptimizerState:
    """State container for L-BFGS optimization."""
    def __init__(self, alpha_true: np.ndarray, rho_true: np.ndarray, M1c: np.ndarray, Bc: np.ndarray):
        self.alpha_true = alpha_true
        self.rho_true = rho_true
        self.M1c = M1c
        self.Bc = Bc
        self.alpha_prev = None
        
    def loss_function(self, alpha_rec: np.ndarray) -> float:
        """Weighted loss with regularization."""
        v_rec = self.Bc @ alpha_rec
        rho_rec, _, _ = cycle_weights(self.M1c, self.Bc, v_rec)
        
        # Primary loss terms
        loss_alpha = 10.0 * norm(alpha_rec - self.alpha_true)**2
        loss_rho = 0.1 * norm(rho_rec - self.rho_true)**2
        
        # Regularization for smoothness
        if self.alpha_prev is not None:
            loss_reg = 0.01 * norm(alpha_rec - self.alpha_prev)**2
        else:
            loss_reg = 0.0
        self.alpha_prev = alpha_rec.copy()
        
        return loss_alpha + loss_rho + loss_reg

def run_feedback(
    Nx: int = 16,
    Ny: int = 16,
    alpha_phi: float = 0.9,
    alpha_n: float = 0.435,
    noise: float = 0.03,
    harmonic_noise_frac: float = 0.2,
    freeze_projector: bool = True,
    steps: int = 20,
    lr: float = 0.5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # --- Coarse reference setup ---
    d0c, d1c, M0c, M1c, M2c = build_dec_mats(Nx, Ny)
    Bc, _ = harmonic_basis(M1c, laplacian_1form(d0c, d1c, M0c, M1c, M2c), k=2)
    v_c = alpha_phi * Bc[:, 0] + alpha_n * Bc[:, 1]
    rho_t, S_t, alpha_t = cycle_weights(M1c, Bc, v_c)

    # Intergrid operators (M-isometry mode)
    P, R = build_prolong_restrict_edges(Nx, Ny, mode="M_isometry")

    # --- Initialize fine grid with biased metric ---
    Nx_f, Ny_f = 2 * Nx, 2 * Ny
    bias_x, bias_y = 1.10, 0.90  # Initial bias
    d0f, d1f, M0f, M1f, M2f = build_dec_mats(Nx_f, Ny_f)
    M1f = apply_metric_bias(M1f, bias_x=bias_x, bias_y=bias_y, Nx=Nx_f, Ny=Ny_f)
    Pif, Bf = transported_projector(M1f, P, Bc)

    # Optimization setup
    opt_state = OptimizerState(alpha_t, rho_t, M1c, Bc)
    v_f_clean = P @ v_c
    eta_f = quasiparticle_noise_hybrid(M1f, v_f_clean.size, noise, rng, Bf, harmonic_noise_frac)

    print("it\tloss\t\talpha_rec\t\tbias_x\tbias_y\tSdr\t\tgap\t\tDK_budget")
    for it in range(steps):
        # --- Current state evaluation ---
        v_c_rec = R @ (Pif @ (v_f_clean + eta_f))
        rho_r, S_r, alpha_r = cycle_weights(M1c, Bc, v_c_rec)
        gap, DK = dk_budget_and_gap(d0c, d1c, M0c, M1c, M2c, d0f, d1f, M0f, M1f, M2f, P, R)

        # --- L-BFGS optimization ---
        def loss_wrapper(bx_by: np.ndarray) -> float:
            """Wrapper for L-BFGS that updates metric and recomputes projection."""
            bx, by = bx_by
            M1f_curr = apply_metric_bias(M1f, bias_x=bx, bias_y=by, Nx=Nx_f, Ny=Ny_f)
            if freeze_projector:
                Pif_curr, Bf_curr = Pif, Bf
            else:
                Pif_curr, Bf_curr = transported_projector(M1f_curr, P, Bc)
            v_rec_curr = R @ (Pif_curr @ (v_f_clean + eta_f))
            _, _, alpha_curr = cycle_weights(M1c, Bc, v_rec_curr)
            return opt_state.loss_function(alpha_curr)

        # Compute gradient using complex-step
        grad = complex_step_gradient(loss_wrapper, np.array([bias_x, bias_y]))

        # L-BFGS step
        result = minimize(
            loss_wrapper,
            np.array([bias_x, bias_y]),
            method='L-BFGS-B',
            jac=lambda x: complex_step_gradient(loss_wrapper, x),
            options={'maxiter': 5, 'ftol': 1e-10}
        )
        bias_x, bias_y = result.x

        # --- Diagnostics ---
        current_loss = result.fun
        Sdr = abs(S_r - S_t)
        print(
            f"{it:02d}\t{current_loss:.3e}\t{np.round(alpha_r,6)}\t{bias_x:.4f}\t{bias_y:.4f}\t"
            f"{Sdr:.2e}\t{gap:.3e}\t{DK:.3e}"
        )

if __name__ == "__main__":
    run_feedback(
        Nx=16,
        Ny=16,
        alpha_phi=0.9,
        alpha_n=0.435,
        noise=0.03,
        harmonic_noise_frac=0.2,
        freeze_projector=True,
        steps=20,
        lr=0.5,
        seed=0,
    )