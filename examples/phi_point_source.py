import argparse
import numpy as np

# --- pull from your package ---
from sfit_sgf.phi.lattice import zeros_field, curl
from sfit_sgf.phi.integrator import integrate  # must accept J and nu
from sfit_sgf.phi.entropy import symbolic_entropy

def gaussian_current(Nx, Ny, a, center=None, sigma=3.0, Jmag=1.0, comp=1):
    """
    Build a smooth localized source J (2-component field) with units of 'current density'.
    comp: which component to use (0->Jx, 1->Jy)
    """
    if center is None:
        center = (0.5*(Nx-1)*a, 0.5*(Ny-1)*a)
    x0, y0 = center

    x = np.arange(Nx)*a
    y = np.arange(Ny)*a
    X, Y = np.meshgrid(x, y, indexing="ij")
    R2 = (X - x0)**2 + (Y - y0)**2
    blob = Jmag * np.exp(-R2 / (2.0*sigma**2))

    J = np.zeros((Nx, Ny, 2), dtype=float)
    J[..., comp] = blob
    return J

def radial_profile(Z, a, center_xy=None, rmax=None, nbins=64):
    """
    Radially average a scalar field Z (Nx,Ny) around center_xy in physical units.
    """
    Nx, Ny = Z.shape
    x = np.arange(Nx)*a
    y = np.arange(Ny)*a
    if center_xy is None:
        center_xy = (0.5*(Nx-1)*a, 0.5*(Ny-1)*a)
    cx, cy = center_xy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).ravel()
    V = Z.ravel()

    if rmax is None:
        rmax = 0.5 * a * np.hypot(Nx-1, Ny-1)

    bins = np.linspace(0.0, rmax, nbins+1)
    idx = np.digitize(R, bins) - 1
    prof = np.zeros(nbins)
    cnt = np.zeros(nbins)
    for k in range(nbins):
        mask = (idx == k)
        if np.any(mask):
            prof[k] = V[mask].mean()
            cnt[k] = mask.sum()

    r = 0.5*(bins[:-1] + bins[1:])
    return r, prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--a", type=float, default=1.0, help="grid spacing [units: e.g., mm]")
    ap.add_argument("--dt", type=float, default=0.1, help="time step")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--nu", type=float, default=1e-2, help="entropy descent strength")
    ap.add_argument("--sigma", type=float, default=3.0, help="source size (in grid units of a)")
    ap.add_argument("--Jmag", type=float, default=1.0, help="source magnitude")
    ap.add_argument("--component", type=int, default=1, help="which J component (0 or 1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- Physical interpretation used here ---
    # 2D square sample, spacing a [e.g., mm]. A = Phi is a 2-component gauge-like field on grid edges.
    # F = curl(A) is the out-of-plane “field strength”. Energy ~ ∫ (1/2) F^2 dA.
    # With nu=0 this reduces to a Maxwell-like Poisson solve: -ΔA ≈ J (in Coulomb gauge).
    # With nu>0 we add -nu * ∇S_U to drive toward coherent patterns (entropy descent).

    Nx, Ny, a = args.Nx, args.Ny, args.a
    rng = np.random.default_rng(args.seed)

    Phi = zeros_field(Nx, Ny)           # shape (Nx,Ny,2)
    J = gaussian_current(Nx, Ny, a, sigma=args.sigma*a, Jmag=args.Jmag, comp=args.component)

    # Diagnostics
    def energy_from_F(F):  # average energy density times area (or just mean)
        return 0.5 * np.mean(F**2)

    # Before
    F0 = curl(Phi, a)        # scalar (Nx,Ny)
    S0, _ = symbolic_entropy(Phi)
    E0 = energy_from_F(F0)
    res0 = np.linalg.norm(F0)   # simple surrogate residual
    print(f"Before: ||F||={res0:.6e}  S={S0:.6f}  E={E0:.6e}")

    # Integrate
    for t in range(args.steps):
        Phi, res = integrate(Phi, dt=args.dt, J=J, nu=args.nu)
        if (t % max(1, args.steps // 10)) == 0 or t == args.steps-1:
            F = curl(Phi, a)
            S, _ = symbolic_entropy(Phi)
            E = energy_from_F(F)
            print(f"t={t:5d}  ||res||={res:.3e}  S={S:.6f}  E={E:.6e}")

    # After
    F = curl(Phi, a)
    S, _ = symbolic_entropy(Phi)
    E = energy_from_F(F)
    print(f"After : ||F||={np.linalg.norm(F):.6e}  S={S:.6f}  E={E:.6e}")

    # Radial decay -> effective mass estimate (fit exponential on tail)
    r, prof = radial_profile(np.abs(F), a, nbins=60)
    # Pick a tail window (skip center), simple log-linear fit:
    mask = (r > 5*a) & (prof > 1e-12)
    if np.count_nonzero(mask) >= 10:
        y = np.log(prof[mask])
        x = r[mask]
        # least squares fit: y = c - m_eff * x
        A = np.vstack([np.ones_like(x), -x]).T
        c, m_eff = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"Estimated effective mass from tail: m_eff ~ {m_eff:.4f} (1/units of length)")
    else:
        print("Tail too small for a robust mass fit; try larger steps or Jmag.")

if __name__ == "__main__":
    main()
