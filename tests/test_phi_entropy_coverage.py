import numpy as np
import pytest

from sfit_sgf.phi.entropy import SemanticEntropy
from sfit_sgf.phi import rect_grid  # just to import package pieces, not used here

def test_prob_normalizes_hard_and_soft():
    bins = np.linspace(-1.0, 1.0, 7)
    Phi = np.linspace(-0.8, 0.9, 25)

    # hard binning
    H = SemanticEntropy(bins, sigma=None)
    p_h = H.prob(Phi)
    assert np.isclose(p_h.sum(), 1.0)
    assert np.all(p_h >= 0)

    # soft binning
    S = SemanticEntropy(bins, sigma=0.2)
    p_s = S.prob(Phi)
    assert np.isclose(p_s.sum(), 1.0)
    assert np.all(p_s >= 0)
    # sanity: typically spreads mass across bins
    assert np.count_nonzero(p_s) >= np.count_nonzero(p_h)

def test_entropy_grad_soft_matches_directional_derivative():
    rng = np.random.default_rng(0)
    bins = np.linspace(-2.0, 2.0, 11)
    Phi = rng.standard_normal(40) * 0.3
    S = SemanticEntropy(bins, sigma=0.3)

    g = S.grad(Phi)
    assert g.shape == Phi.shape
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 0

    # directional derivative check
    u = rng.standard_normal(Phi.size)
    u /= np.linalg.norm(u)
    eps = 1e-4
    Sd = S.entropy(Phi + eps * u) - S.entropy(Phi - eps * u)
    dir_fd = Sd / (2 * eps)
    dir_ad = float(np.dot(g, u))
    assert np.isfinite(dir_fd)
    # loose tolerance—just verifying we hit the right branch and math is sane
    assert np.isclose(dir_ad, dir_fd, rtol=5e-2, atol=5e-3)

def test_entropy_grad_hard_is_zero():
    bins = np.linspace(-1, 1, 5)
    Phi = np.linspace(-0.9, 0.9, 20)
    H = SemanticEntropy(bins, sigma=None)
    g = H.grad(Phi)
    assert np.allclose(g, 0.0)

def test_prob_empty_and_nan_fallbacks():
    bins = np.linspace(-1, 1, 5)

    # empty Phi -> zeros vector (sum=0) but entropy() should be well-defined (0)
    S = SemanticEntropy(bins, sigma=None)
    p_empty = S.prob(np.array([]))
    assert p_empty.shape == bins.shape
    assert np.allclose(p_empty, 0.0)
    assert S.entropy([]) == 0.0

    # soft path + NaN -> not finite total -> uniform fallback
    S2 = SemanticEntropy(bins, sigma=0.2)
    p_nan = S2.prob(np.array([np.nan]))
    assert np.allclose(p_nan, np.full(bins.size, 1.0 / bins.size))
