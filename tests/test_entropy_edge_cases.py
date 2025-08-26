import numpy as np
from ._helpers import shannon_entropy

def test_entropy_invariance_to_scaling():
    w = np.array([3.0, 1.5, 0.5])  # proportional to [0.6, 0.3, 0.1]
    S1 = shannon_entropy(w)
    S2 = shannon_entropy(10.0 * w)  # scaled by constant
    assert abs(S1 - S2) < 1e-12

def test_entropy_handles_zeros():
    S = shannon_entropy(np.array([1.0, 0.0, 0.0]))
    assert abs(S - 0.0) < 1e-12

def test_uniform_vs_peaked():
    u = np.ones(4) / 4.0
    S_u = shannon_entropy(u)  # = ln 4
    p = np.array([1.0, 0.0, 0.0, 0.0])
    S_p = shannon_entropy(p)
    assert S_u > S_p
    assert np.allclose(S_u, np.log(4.0))
