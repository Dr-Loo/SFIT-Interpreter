import numpy as np
from sfit_sgf.phi import zeros_field, PhiState, step_until, symbolic_entropy
from sfit_sgf.phi.integrator import integrate

def make_point_source(N=32, Jmag=1.0):
    J = zeros_field(N, N)
    J[N//2, N//2, 0] = Jmag
    return J

def run_once(N=32, steps=200, dt=0.1, nu=0.0, Jmag=1.0, nbins=8):
    Phi = zeros_field(N, N)
    state = PhiState(Phi=Phi, a=1.0)
    J = make_point_source(N, Jmag)

    S0, _ = symbolic_entropy(state.Phi, nbins=nbins)
    res = float("inf")
    for _ in range(steps):
        state.Phi, res = integrate(state.Phi, dt=dt, J=J, nu=nu, a=state.a, nbins=nbins)
    S1, _ = symbolic_entropy(state.Phi, nbins=nbins)
    return res, S0, S1

def main():
    # Maxwell-like: residual should drop well below initial scale
    res0, S0a, S1a = run_once(nu=0.0)
    print(f"[nu=0] residual: {res0:.3e}, S0={S0a:.6f}, S1={S1a:.6f}")
    assert res0 < 2.0, "Residual did not decrease enough for nu=0"

    # Entropy-descent: symbolic entropy should decrease significantly
    res1, S0b, S1b = run_once(nu=1e-2)
    print(f"[nu>0] residual: {res1:.3e}, S0={S0b:.6f}, S1={S1b:.6f}")
    assert S1b < 0.5, "Symbolic entropy did not collapse under entropy descent"

if __name__ == "__main__":
    main()
