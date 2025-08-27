import numpy as np
from sfit_sgf.phi import rect_grid, SemanticEntropy, PhiState, step_until

def test_entropy_hard_and_soft_bins():
    inc, h = rect_grid(6, 6)
    n_edges = inc.D2.shape[1]
    n_faces = inc.D2.shape[0]

    # Hard-binning path (sigma=None / <=0)
    Phi = np.zeros(n_edges)
    st = PhiState(
        inc, h, Phi,
        J=np.zeros(n_faces),
        entropy=SemanticEntropy(np.linspace(-1, 1, 5), sigma=None),
        nu=0.0, dt=0.05,
    )
    r0 = np.linalg.norm(st.residual())
    st.step(3)                # exercise method stepping
    r1 = np.linalg.norm(st.residual())
    assert np.isfinite(r0) and np.isfinite(r1)

    # Soft (Gaussian) kernel path (sigma>0), nontrivial Phi
    rng = np.random.default_rng(42)
    Phi2 = 0.1 * rng.standard_normal(n_edges)
    st2 = PhiState(
        inc, h, Phi2,
        J=np.zeros(n_faces),
        entropy=SemanticEntropy(np.linspace(-0.5, 0.5, 7), sigma=0.2),
        nu=0.1, dt=0.05,
    )
    # residual calls entropy.grad internally
    r2 = np.linalg.norm(st2.residual())
    st2.step(2)
    r3 = np.linalg.norm(st2.residual())
    assert np.isfinite(r2) and np.isfinite(r3)

def test_step_until_branches():
    inc, h = rect_grid(6, 6)
    n_edges = inc.D2.shape[1]
    n_faces = inc.D2.shape[0]
    rng = np.random.default_rng(0)
    Phi = 0.05 * rng.standard_normal(n_edges)

    # Use nu=0: pure residual descent (no entropy term)
    st = PhiState(inc, h, Phi, J=np.zeros(n_faces), entropy=None, nu=0.0, dt=0.1)

    # Early-stop branch: huge tolerance (should break almost immediately)
    st_early = step_until(st, iters=5, tol=1e6)
    assert isinstance(st_early, PhiState)

    # Full-iteration branch: tiny tolerance (runs all iters)
    st2 = PhiState(inc, h, Phi.copy(), J=np.zeros(n_faces), entropy=None, nu=0.0, dt=0.05)
    st_full = step_until(st2, iters=5, tol=0.0)
    assert isinstance(st_full, PhiState)
