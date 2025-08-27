import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
m_H = 125  # GeV
beta = 0.01
Lambda_torsion = 2.49  # GeV
grid_size = 256
steps = 1000

# Initialize fields
N_defect = np.exp(-np.random.rand(grid_size, grid_size))  # ∇N_defect ~ exp(-m/20)
phi_lock = np.log(1 + m_H * np.ones((grid_size, grid_size)))
E_curv = 0.01 * (np.random.randn(grid_size, grid_size)**2)  # Curvature energy

# Stochastic evolution
def evolve_phi(phi, D=1.0, dt=0.1):
    noise = np.random.randn(*phi.shape) * np.sqrt(dt)
    return phi + D * gaussian_filter(phi, sigma=1) * dt + noise

# Simulate fragmentation
phi_shell = N_defect * Lambda_torsion * phi_lock * np.exp(-beta * E_curv)
fragmentation_history = []

for _ in range(steps):
    phi_shell = evolve_phi(phi_shell)
    if np.var(phi_shell) > 1.0:  # Domain formation threshold
        fragmentation_history.append(phi_shell.copy())

# Plot last fragmentation snapshot
plt.imshow(fragmentation_history[-1], cmap='viridis')
plt.title("Fragmented $\\phi$-lock Domains")
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()
