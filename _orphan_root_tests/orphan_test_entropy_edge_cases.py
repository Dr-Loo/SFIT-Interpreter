import numpy as np

def shannon_entropy(rho, eps=1e-16):
    rho = np.asarray(rho, dtype=float)
    rho = np.clip(rho, 0.0, 1.0)
    s = rho.sum()
    if s == 0:
        return 0.0
    rho = rho / s
    return float(-np.sum(rho * np.log(rho + eps)))

def test_entropy_one_hot_is_zero():
    rho = np.array([1.0, 0.0, 0.0])
    assert abs(shannon_entropy(rho) - 0.0) < 1e-12

def test_entropy_uniform_is_log_n():
    for n in (2, 3, 10, 50):
        rho = np.ones(n) / n
        S = shannon_entropy(rho)
        assert abs(S - np.log(n)) < 1e-12

def test_entropy_near_zero_bins_stable():
    rho = np.array([0.9999999, 1e-7, 1e-12, 0.0])
    S = shannon_entropy(rho)
    # Finite and small
    assert np.isfinite(S)
    assert S < 1e-3

def test_entropy_invariance_to_scaling():
    # If someone passes unnormalized weights, entropy should be the same after normalization
    w = np.array([3.0, 1.5, 0.5])      # proportional to [0.6, 0.3, 0.1]
    rho = w / w.sum()
    S1 = shannon_entropy(rho)
    S2 = shannon_entropy(10.0 * w)     # scaled by constant
    assert abs(S1 - S2) < 1e-12
