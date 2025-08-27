import numpy as np
from dataclasses import dataclass
from .lattice import Incidence, Hodge, curl, divergence
from .entropy import SemanticEntropy


@dataclass
class PhiState:
    inc: Incidence
    h: Hodge
    Phi: np.ndarray          # edges
    J: np.ndarray            # faces (source)
    entropy: SemanticEntropy | None
    nu: float = 0.0
    dt: float = 0.1

    # ---- instance helpers expected by tests ----
    def residual(self) -> np.ndarray:
        return residual(self)

    def residual_norm(self) -> float:
        return residual_norm(self)

    def step(self, n: int = 1) -> "PhiState":
        """Advance the state by n Euler steps (default: 1)."""
        for _ in range(int(n)):
            step(self)
        return self



# ---- functional API (used by the instance wrappers) ----

def residual(state: PhiState) -> np.ndarray:
    """Face residual r = curl(Phi) - J."""
    return curl(state.inc, state.Phi) - state.J


def residual_norm(state: PhiState) -> float:
    return float(np.linalg.norm(residual(state)))


def step(state: PhiState) -> PhiState:
    r_faces = residual(state)
    g = divergence(state.inc, r_faces)  # back to edges
    if state.entropy is not None and state.nu != 0.0:
        g = g + state.nu * state.entropy.grad(state.Phi)
    state.Phi = state.Phi - state.dt * g
    return state


def step_until(state: PhiState, iters: int = 200, tol: float = 1e-6) -> PhiState:
    for _ in range(int(iters)):
        if residual_norm(state) <= tol:
            break
        step(state)
    return state


__all__ = ["PhiState", "residual", "residual_norm", "step", "step_until"]
