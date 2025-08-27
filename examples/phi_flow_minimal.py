import numpy as np
from sfit_sgf.phi import rect_grid, SemanticEntropy, PhiState, step_until

def main():
    nx, ny = 16, 16
    inc, h = rect_grid(nx, ny)
    n_edges = inc.D2.shape[1]
    n_faces = inc.D2.shape[0]

    rng = np.random.default_rng(0)
    Phi0 = 0.1 * rng.standard_normal(n_edges)
    J = np.zeros(n_faces)

    # smooth sector histogram over |F|
    centers = np.linspace(0.0, 1.0, 8)
    entropy = SemanticEntropy(centers=centers, sigma=0.25)

    st = PhiState(inc, h, Phi0, J=J, entropy=entropy, nu=0.1, dt=5e-2)

    print("||residual|| before:", np.linalg.norm(st.residual()))
    step_until(st, iters=2000, tol=1e-6)
    print("||residual|| after :", np.linalg.norm(st.residual()))

if __name__ == "__main__":
    main()
