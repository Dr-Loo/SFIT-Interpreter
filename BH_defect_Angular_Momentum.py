import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# ========================
# 1. STABLE KERR DEFECT SETUP
# ========================
N = 128
x, y = np.indices((N, N)) - N//2
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Physical parameters (normalized units)
M = 0.1          # Reduced mass for stability
a = 0.5          # Spin parameter
r_s = 15         # Horizon radius
ergo_width = 3   # Tighter ergoregion

# Stable defect formulation
rho_D = M * np.exp(-(r - r_s)**2 / (2*ergo_width**2))
angular_term = 0.3 * a * np.sin(2*theta) * np.exp(-(r - r_s)**2 / (4*ergo_width**2)) 
rho_D = np.clip(rho_D + angular_term, 0, 1)  # Enforce bounds

# ========================
# 2. STABILIZED SOLVER
# ========================
phi = np.zeros_like(rho_D)
for _ in range(100):
    # Modified with damping term (0.1*∇²ϕ) for stability
    phi += 0.01 * (0.1*laplace(phi, mode='reflect') + rho_D)
    phi = np.clip(phi, 0, 2*M)  # Physical curvature bound

# ========================
# 3. ANALYSIS
# ========================
max_phi = np.max(phi)
max_r = np.where(phi == max_phi)[0][0] - N//2

print(f"""
=== STABLE KERR DEFECT ===
1. Max ϕ: {max_phi:.3f} (at r={max_r})
2. Spin Parameter: {a}
3. Ergosphere: r={r_s}±{ergo_width}px
4. Curvature Bounds: [0, {2*M:.1f}] (enforced)
""")

# ========================
# 4. VISUALIZATION
# ========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

im1 = ax1.imshow(rho_D, cmap='plasma')
ax1.set_title(f'Kerr Defect (a={a})')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(phi, cmap='viridis', vmax=2*M)
ax2.set_title(f'Stable Emergent ϕ (Max={max_phi:.2f} at r={max_r})')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()