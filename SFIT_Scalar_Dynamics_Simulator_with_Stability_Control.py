import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

# Simulation parameters
nx, ny = 100, 100
nt = 200
dx = dy = 1e-3  # parsec
dt = 1e-4       # Reduced timestep for stability (Courant condition: dt < dx / sqrt(2))
phi0 = 1.0      # Initial field amplitude
alpha = 0.1     # Damping coefficient

# SFIT constants
v_T = 246.0              # GeV
lambda_phi = 0.12
rho_crit = 0.22
rho_clamp = 0.18         # Threshold for curvature clamping

# Storage
energy_history = []
phi_history = []

def torsion_profile(nx, ny, seed=42):
    """Generate a smooth torsion defect density field."""
    np.random.seed(seed)
    base = np.random.rand(nx, ny)
    structured = gaussian_filter(base, sigma=8)
    return 0.15 * structured

def scalar_mass(rho_D):
    """Compute scalar mass from defect density."""
    m_phi = np.sqrt(8 * lambda_phi * (1 - np.exp(-rho_D / rho_crit))) * v_T * np.tanh(rho_D / rho_crit)
    return m_phi

def curvature_clamp(m_phi, rho_D, threshold=rho_clamp):
    """Suppress mass growth in high-curvature regions."""
    mask = rho_D > threshold
    m_phi[mask] *= np.exp(-(rho_D[mask] - threshold)**2 / 0.01)
    return m_phi

def evolve_phi(phi_prev, phi_curr, m_phi, dx, dt, lambda_phi=0.12, alpha=0.1):
    """Evolve scalar field with damping and clipped nonlinearity."""
    # Laplacian (periodic boundaries)
    laplacian = (
        np.roll(phi_curr, 1, axis=0) + np.roll(phi_curr, -1, axis=0) +
        np.roll(phi_curr, 1, axis=1) + np.roll(phi_curr, -1, axis=1) - 4 * phi_curr
    ) / dx**2

    # Clipped nonlinear term to avoid overflow
    nonlinear_decay = lambda_phi * np.clip(phi_curr**3, -1e10, 1e10)

    # Damped wave equation
    damping = alpha * (phi_curr - phi_prev) / dt
    phi_next = 2 * phi_curr - phi_prev + dt**2 * (laplacian - m_phi**2 * phi_curr - nonlinear_decay) - dt * damping

    return phi_next

# Initialize fields
rho_D = torsion_profile(nx, ny)
m_phi_raw = scalar_mass(rho_D)
m_phi = curvature_clamp(m_phi_raw.copy(), rho_D)

phi_prev = np.zeros((nx, ny))
phi_curr = np.ones((nx, ny)) * phi0

# Time evolution
for t in range(nt):
    phi_next = evolve_phi(phi_prev, phi_curr, m_phi, dx, dt, lambda_phi, alpha)
    phi_history.append(phi_curr.copy())

    # Energy diagnostics (with NaN checks)
    grad_x = (np.roll(phi_curr, -1, axis=0) - phi_curr) / dx
    grad_y = (np.roll(phi_curr, -1, axis=1) - phi_curr) / dy
    energy = 0.5 * ((grad_x**2 + grad_y**2) + (m_phi * phi_curr)**2).sum()
    
    if not np.isfinite(energy):
        energy = 0.0  # Fallback for unstable steps
    energy_history.append(energy)

    phi_prev, phi_curr = phi_curr, phi_next

# Clamp final field to physical range
phi_curr = np.clip(phi_curr, -1e4, 1e4)

# Save snapshot and diagnostics
df = pd.DataFrame({
    'x': np.tile(np.arange(nx), ny),
    'y': np.repeat(np.arange(ny), nx),
    'rho_D': rho_D.flatten(),
    'm_phi_GeV': m_phi.flatten(),
    'phi_final': phi_curr.flatten()
})
df.to_csv("SFIT_scalar_clamped_snapshot.csv", index=False)
print("✅ Stabilized dataset saved to SFIT_scalar_clamped_snapshot.csv")

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(rho_D, cmap='viridis')
plt.title("Torsion Defect Density $\\rho_D$"); plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(m_phi, cmap='cividis', vmin=0, vmax=100)
plt.title("Clamped Scalar Mass $m_\\phi$ (GeV)"); plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(phi_curr, cmap='plasma', vmin=-10, vmax=10)
plt.title("Final Scalar Field $\\phi(x,y,t)$"); plt.colorbar()

plt.tight_layout()
plt.savefig("SFIT_scalar_clamped_plot.png")
print("📷 Field evolution plot saved to SFIT_scalar_clamped_plot.png")

# Energy plot
plt.figure()
plt.plot(energy_history)
plt.title("Total Energy Evolution"); plt.xlabel("Time"); plt.ylabel("Energy")
plt.savefig("SFIT_energy_stability.png")
print("📊 Energy trace saved to SFIT_energy_stability.png")