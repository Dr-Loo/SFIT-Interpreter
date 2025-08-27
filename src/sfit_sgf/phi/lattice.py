import numpy as np

def rect_grid(nx: int, ny: int):
    """2D rectangular grid (unit spacing). Returns counts and orientation info."""
    assert nx >= 2 and ny >= 2
    n_nodes = nx * ny
    # Edges: horizontal (nx-1)*ny + vertical nx*(ny-1)
    n_e_h = (nx - 1) * ny
    n_e_v = nx * (ny - 1)
    n_edges = n_e_h + n_e_v
    n_faces = (nx - 1) * (ny - 1)

    # Build incidence D2: edges->faces (shape: n_faces x n_edges)
    # Face orientation: ccw loop (right, up, left, down).
    D2 = np.zeros((n_faces, n_edges), dtype=float)

    def e_h(i, j):  # edge idx for horizontal edge from (i,j)->(i+1,j)
        return j * (nx - 1) + i
    def e_v(i, j):  # vertical edge from (i,j)->(i,j+1)
        return n_e_h + i * (ny - 1) + j

    f = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            # right: (i,j)->(i+1,j)
            D2[f, e_h(i, j)] = +1.0
            # up: (i+1,j)->(i+1,j+1) vertical edge but orientation upward
            D2[f, e_v(i+1, j)] = +1.0
            # left: (i+1,j+1)->(i,j+1) is the negative of (i,j+1)->(i+1,j+1)
            D2[f, e_h(i, j+1)] = -1.0
            # down: (i,j+1)->(i,j) is the negative of (i,j)->(i,j+1)
            D2[f, e_v(i, j)] = -1.0
            f += 1

    # Simple diagonal Hodge stars (unit cells): *1 on edges, *1 on faces
    star1 = np.ones(n_edges, dtype=float)   # |dual edge| ~ 1
    star2 = np.ones(n_faces, dtype=float)   # |dual face| ~ 1

    return Incidence(D2=D2), Hodge(star1=star1, star2=star2)

class Incidence:
    def __init__(self, D2: np.ndarray):
        self.D2 = D2  # edges -> faces

class Hodge:
    def __init__(self, star1: np.ndarray, star2: np.ndarray):
        self.star1 = star1
        self.star2 = star2
