import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

# Simulation parameters
nx, ny = 100, 100          # grid size
nt = 200                   # time steps
dx = dy = 1e-3             # parsec spacing
dt = 1e-3                  # time step (arbitrary units)
phi0 = 1.0                 # initial field amplitude

# SFIT constants
v_T = 246.0                # GeV
lambda_phi = 0.12
rho_crit = 0.22

def torsion_profile(nx, ny, seed=42):
    """Generate anisotropic rho_D grid with spatial structure"""
    np.random.seed(seed)
    base = np.random.rand(nx, ny)
    structured = gaussian_filter(base, sigma=8)
    return 0.15 * structured  # max rho_D ~ 0.15

def scalar_mass(rho_D):
    """SFIT scalar mass from defect density"""
    m_phi = np.sqrt(8 * lambda_phi * (1 - np.exp(-rho_D / rho_crit))) * v_T * np.tanh(rho_D / rho_crit)
    return m_phi  # GeV

def evolve_phi(phi_prev, phi_curr, m_phi, dx, dt):
    """Scalar field evolution with position-dependent mass"""
    laplacian = (
        np.roll(phi_curr, 1, axis=0) + np.roll(phi_curr, -1, axis=0) +
        np.roll(phi_curr, 1, axis=1) + np.roll(phi_curr, -1, axis=1) - 4 * phi_curr
    ) / dx**2
    return 2 * phi_curr - phi_prev + dt**2 * (laplacian - m_phi**2 * phi_curr)

# Initialize fields
rho_D = torsion_profile(nx, ny)
m_phi = scalar_mass(rho_D)

phi_prev = np.zeros((nx, ny))
phi_curr = np.ones((nx, ny)) * phi0
phi_history = []

for t in range(nt):
    phi_next = evolve_phi(phi_prev, phi_curr, m_phi, dx, dt)
    phi_history.append(phi_curr.copy())
    phi_prev, phi_curr = phi_curr, phi_next

# Export scalar field snapshot and torsion profile
df = pd.DataFrame({
    'x': np.tile(np.arange(nx), ny),
    'y': np.repeat(np.arange(ny), nx),
    'rho_D': rho_D.flatten(),
    'm_phi_GeV': m_phi.flatten(),
    'phi_final': phi_curr.flatten()
})
df.to_csv("SFIT_scalar_evolution_snapshot.csv", index=False)
print("✅ Snapshot saved to SFIT_scalar_evolution_snapshot.csv")

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rho_D, cmap='viridis')
plt.title("Torsion Defect Density $\\rho_D(x,y)$")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(phi_curr, cmap='plasma')
plt.title("Scalar Field Amplitude $\\phi(x,y,t_{final})$")
plt.colorbar()
plt.tight_layout()
plt.savefig("SFIT_scalar_evolution.png")
print("📷 Scalar field evolution plot saved to SFIT_scalar_evolution.png")
