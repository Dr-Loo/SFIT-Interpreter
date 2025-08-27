import numpy as np
from .lattice import Incidence, Hodge
from .entropy import SemanticEntropy

class PhiState:
    """
    Minimal U(1) Φ-field state on a fixed lattice.
    Φ: (n_edges,), F = D2 @ Φ (on faces).
    J: (n_faces,) source term in the equation.
    """
    def __init__(self, inc: Incidence, h: Hodge, Phi, J=None, entropy: SemanticEntropy=None, nu=0.0, dt=1e-2):
        self.D2 = inc.D2
        self.star1 = h.star1
        self.star2 = h.star2
        self.Phi = np.array(Phi, dtype=float)
        self.nu = float(nu)
        self.dt = float(dt)
        self.J = np.zeros(self.D2.shape[0], dtype=float) if J is None else np.array(J, dtype=float)
        self.entropy = entropy

    def curvature(self):
        return self.D2 @ self.Phi  # F on faces

    def residual(self):
        """
        r = d*F - *J + nu * ∇S_U  (all on faces for this abelian prototype)
        We compute d*F as D2 @ Φ mapped to faces and scaled by star2 if needed.
        Here we keep * as identity (star2=1) for simplicity.
        """
        F = self.curvature()
        r = F - self.J  # since * = I and d*F ~ F in this simple setup
        if self.entropy and self.nu != 0.0:
            _, dS_dF = self.entropy.S_and_gradF(F)
            r = r + self.nu * dS_dF
        return r

    def grad_Phi(self):
        """
        ∇_Φ (1/2 ||r||^2 ) = (∂r/∂Φ)^T r.
        r = F - J + nu dS/dF, F = D2 Φ.
        ∂r/∂Φ = D2 + nu * (∂/∂Φ)(dS/dF) ~ D2 if we neglect second derivative for a stable first-order scheme.
        """
        r = self.residual()
        return self.D2.T @ r  # Gauss-Newton / first-order

    def step(self, n=1):
        for _ in range(n):
            g = self.grad_Phi()
            self.Phi = self.Phi - self.dt * g

def step_until(state: PhiState, iters=1000, tol=1e-6):
    for _ in range(iters):
        r = state.residual()
        if np.linalg.norm(r) < tol:
            break
        state.step(1)
    return state
