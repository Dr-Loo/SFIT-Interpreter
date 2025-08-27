import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physics parameters
grid_size = 100
time_steps = 4
v_EW = 86.0
rho_D_crit = 0.15

def initialize_defects():
    """Create initial defect configuration with proper parentheses"""
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    relic1 = np.exp(-((x + 0.3)**2 + (y - 0.2)**2) / 0.1)
    relic2 = np.exp(-((x - 0.4)**2 + (y + 0.1)**2) / 0.1)
    return 0.3 * (relic1 + relic2)

def evolve_defects(step, bridge_intensity=0.15):
    """Improved evolution with dynamic bridge intensity"""
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    drift = 0.2 * (step / time_steps)
    
    # Main relics
    relic1 = 0.3 * np.exp(-((x + 0.3 + drift)**2 + (y - 0.2)**2) / 0.1)
    relic2 = 0.3 * np.exp(-((x - 0.4 - drift)**2 + (y + 0.1)**2) / 0.1)
    
    # Dynamic bridge based on step progression
    bridge = bridge_intensity * (1 - 0.1*step) * np.exp(-(x**2 + (y - 0.05)**2) / 0.3)
    
    return np.clip(relic1 + relic2 + bridge, 0, 0.35)

# Create figure with professional layout
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, time_steps + 1, figure=fig, width_ratios=[1]*time_steps + [0.05])
cbar_ax = fig.add_subplot(gs[-1])  # Dedicated colorbar axis

# Generate time sequence
images = []
for t in range(time_steps):
    ax = fig.add_subplot(gs[t])
    rho_D = evolve_defects(t)  # Simplified call
    m_phi = np.sqrt(0.8 * (1 - np.exp(-rho_D / 0.1))) * v_EW
    m_phi_masked = np.ma.masked_where(rho_D < rho_D_crit, m_phi)
    
    im = ax.imshow(m_phi_masked, cmap='inferno', vmin=60, vmax=90)
    ax.set_title(f"t = {t + 1}")
    ax.set_xlabel(f"Flux = {0.15 + 0.02*t:.2f}")
    images.append(im)

# Final touches
fig.colorbar(images[0], cax=cbar_ax, label=r'Scalar mass $m_\phi$ (GeV)')
fig.suptitle("Theorem 5: Global Scalar Locking via Topological Flux Continuity", y=1.05)
plt.savefig("Theorem5_GlobalLocking.png", dpi=300, bbox_inches='tight')
plt.show()