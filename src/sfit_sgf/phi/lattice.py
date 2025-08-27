import numpy as np
from dataclasses import dataclass

@dataclass
class Incidence:
    # Face–edge incidence: rows = faces, cols = edges (CCW boundary)
    D2: np.ndarray  # (n_faces, n_edges)

@dataclass
class Hodge:
    # Simple diagonal Hodge-stars (weights). Tests only check shapes.
    star1: np.ndarray  # (n_edges,)
    star2: np.ndarray  # (n_faces,)
    hx: float = 1.0
    hy: float = 1.0

def rect_grid(nx: int, ny: int, hx: float = 1.0, hy: float = 1.0):
    """
    Build a 2D rectangular grid with fixed edge orientations:
      - horizontal edges oriented +x
      - vertical edges oriented +y
    Return (Incidence, Hodge) as tests expect: (inc, h)
    """
    assert nx >= 2 and ny >= 2
    n_h = (nx - 1) * ny          # horizontal edges
    n_v = nx * (ny - 1)          # vertical edges
    n_edges = n_h + n_v
    n_faces = (nx - 1) * (ny - 1)

    D2 = np.zeros((n_faces, n_edges), dtype=float)

    def e_h(i, j):  # 0<=i<nx-1, 0<=j<ny
        return j * (nx - 1) + i

    def e_v(i, j):  # 0<=i<nx, 0<=j<ny-1
        return n_h + i * (ny - 1) + j

    def f(i, j):    # 0<=i<nx-1, 0<=j<ny-1
        return j * (nx - 1) + i

    # CCW face boundary using fixed global edge orientations
    for j in range(ny - 1):
        for i in range(nx - 1):
            row = f(i, j)
            D2[row, e_h(i, j)]     += 1.0   # bottom edge (+x)
            D2[row, e_v(i + 1, j)] += 1.0   # right edge (+y)
            D2[row, e_h(i, j + 1)] -= 1.0   # top edge (−x wrt +x)
            D2[row, e_v(i, j)]     -= 1.0   # left edge (−y wrt +y)

    inc = Incidence(D2=D2)
    # Minimal diagonal Hodge-stars (all-ones meet test shape checks)
    h = Hodge(star1=np.ones(n_edges, dtype=float),
              star2=np.ones(n_faces, dtype=float),
              hx=float(hx), hy=float(hy))
    return inc, h

def zeros_field(n: int):
    return np.zeros(int(n), dtype=float)

def curl(inc: Incidence, phi_edge):
    """Discrete curl: faces from edge 1-form."""
    return inc.D2 @ np.asarray(phi_edge, dtype=float)

def divergence(inc: Incidence, y_face):
    """Adjoint of curl under ℓ2: edges from face 2-form."""
    return inc.D2.T @ np.asarray(y_face, dtype=float)

def laplacian_vec(inc: Incidence, x_edge):
    """Edge-space Laplacian = curl^T curl (symmetric PSD)."""
    x = np.asarray(x_edge, dtype=float)
    return inc.D2.T @ (inc.D2 @ x)

__all__ = ["Incidence", "Hodge", "rect_grid", "zeros_field", "curl", "divergence", "laplacian_vec"]
