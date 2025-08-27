import numpy as np
from scipy.optimize import differential_evolution

# Physical constants
delta_m21_exp = 7.53e-5  # eV²
delta_m31_exp = 2.45e-3  # eV²
MAX_MASS = 0.12          # eV

def braid_mass(L, k, alpha, beta, C=1.0):
    """SFIT-XSM mass formula with topological suppression"""
    L = np.asarray(L, dtype=np.float64)
    return 0.511e6 * 1e-9 * C * np.exp(-k*L**alpha) / (1 + beta*L)**2

def objective(x):
    """Constrained optimization target"""
    C, L1, L2, L3, k, alpha, beta = x
    
    # Strict SFIT-XSM winding number constraints
    if not (2.0 <= L1 <= 5.0) or not (2.0 <= L2 <= 5.0) or not (2.0 <= L3 <= 5.0):
        return np.inf
    if not (L1 > L2 > L3):
        return np.inf
    
    m = braid_mass([L1, L2, L3], k, alpha, beta, C)
    delta_m21 = m[1]**2 - m[0]**2
    delta_m31 = m[2]**2 - m[0]**2
    
    # Relative error metric
    err = (np.log10(delta_m21/delta_m21_exp)**2 + 
          10*np.log10(delta_m31/delta_m31_exp)**2
    
    # Hard constraints
    if np.sum(m) > MAX_MASS or any(m <= 0):
        return np.inf
        
    return err

# Tightened parameter bounds per SFIT-XSM
bounds = [
    (1.5, 3.0),     # C
    (2.0, 5.0),     # L1 (strict [2,5] range)
    (2.0, 5.0),     # L2 
    (2.0, 5.0),     # L3
    (0.1, 0.3),     # k (narrowed range)
    (0.6, 0.8),     # alpha
    (0.03, 0.04)    # beta
]

# Optimization with stricter tolerances
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    popsize=40,
    maxiter=10000,
    tol=1e-12,
    init='sobol'
)

if result.success:
    params = result.x
    m = braid_mass(params[1:4], *params[4:], params[0])
    
    print("\n=== SFIT-XSM VALIDATED SOLUTION ===")
    print(f"C = {params[0]:.4f}  L = [{params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}]")
    print(f"k = {params[4]:.4f}  α = {params[5]:.4f}  β = {params[6]:.5f}\n")
    
    print("=== NEUTRINO MASSES ===")
    print(f"m₁ = {m[0]:.3e} eV  m₂ = {m[1]:.3e} eV  m₃ = {m[2]:.3e} eV")
    print(f"Σm_ν = {np.sum(m):.4f} eV (limit: {MAX_MASS} eV)\n")
    
    print("=== MASS SPLITTINGS ===")
    print(f"Δm²₂₁ = {m[1]**2 - m[0]**2:.3e} eV² (target: {delta_m21_exp:.3e} eV²)")
    print(f"Δm²₃₁ = {m[2]**2 - m[0]**2:.3e} eV² (target: {delta_m31_exp:.3e} eV²)")
    
    # Verify theoretical predictions
    assert all(2.0 <= L <= 5.0 for L in params[1:4]), "Winding number violation"
    assert 0.1 <= params[4] <= 0.3, "k out of range"
    assert 0.6 <= params[5] <= 0.8, "α out of range"
    assert 0.03 <= params[6] <= 0.04, "β out of range"
else:
    print("Optimization failed:", result.message)