import numpy as np
from sfit_sgf.phi import rect_grid, SemanticEntropy, PhiState, step_until

def test_rect_grid_shapes():
    inc, h = rect_grid(8, 8)
    n_faces, n_edges = inc.D2.shape
    assert n_faces == (8-1)*(8-1)
    assert n_edges == (8-1)*8 + 8*(8-1)
    assert h.star1.shape == (n_edges,)
    assert h.star2.shape == (n_faces,)

def test_entropy_grad_finite():
    inc, h = rect_grid(8, 8)
    n_edges = inc.D2.shape[1]
    Phi = np.zeros(n_edges)
    st = PhiState(
        inc, h, Phi,
        J=np.zeros(inc.D2.shape[0]),
        entropy=SemanticEntropy(np.linspace(0, 1, 6), sigma=0.2),
        nu=0.1, dt=0.05,
    )
    r = st.residual()
    assert np.isfinite(r).all()

def test_flow_residual_decreases_no_entropy():
    # Use nu=0 to test pure residual descent (stable & deterministic)
    inc, h = rect_grid(10, 10)
    n_edges = inc.D2.shape[1]
    n_faces = inc.D2.shape[0]
    rng = np.random.default_rng(1)
    Phi = 0.1 * rng.standard_normal(n_edges)
    st = PhiState(inc, h, Phi, J=np.zeros(n_faces), entropy=None, nu=0.0, dt=0.1)
    r0 = np.linalg.norm(st.residual())
    st.step(50)
    r1 = np.linalg.norm(st.residual())
    assert r1 <= r0 + 1e-9
