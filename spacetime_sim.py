import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import trapezoid

# ========================
# 1. SETUP: ASYMMETRIC DEFECTS
# ========================
N = 128
rho_D = np.zeros((N, N))

# Asymmetric defect configuration
rho_D[N//2-10:N//2+10, N//2-5:N//2+15] = 1.0  # Elliptical defect
rho_D[N//3:N//3+20, 2*N//3:2*N//3+20] = 0.7   # Off-center instanton
rho_D += 0.05 * np.random.rand(N, N)           # Quantum foam

# ==============================
# 2. TOPOLOGICAL SOURCE TERMS
# ==============================
def black_hole_term(r, r_s=15):
    return np.exp(-(r - r_s)**2 / 2)

r = np.linalg.norm(np.indices((N, N)) - N//2, axis=0)  # Fixed parenthesis
H_BH = black_hole_term(r)

# ==============================
# 3. SOLVE EMERGENT GEOMETRY
# ==============================
phi = np.zeros_like(rho_D)
for _ in range(500):
    phi += 0.01 * (laplace(phi, mode='wrap') + rho_D + 0.5*H_BH)

# ==============================
# 4. VISUALIZATION
# ==============================
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(rho_D, cmap='plasma')
plt.title('Defect Density (ρ_D)')

plt.subplot(122)
plt.imshow(phi, cmap='viridis')
plt.title('Emergent Geometry (ϕ)')
plt.show()