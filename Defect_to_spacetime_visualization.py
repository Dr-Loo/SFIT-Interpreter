import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# 1. Set up asymmetric defect lattice
N = 128
rho_D = np.zeros((N, N))
rho_D[N//2-10:N//2+10, N//2-5:N//2+15] = 1.0  # Central defect
rho_D += 0.05 * np.random.rand(N, N)           # Quantum noise

# 2. Solve emergent geometry (∇²ϕ = ρ_D)
phi = np.zeros_like(rho_D)
for _ in range(500):
    phi += 0.01 * (laplace(phi, mode='wrap') + rho_D)

# 3. Visualize
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(rho_D, cmap='plasma'), plt.title('Defect Density')
plt.subplot(122), plt.imshow(phi, cmap='viridis'), plt.title('Emergent Spacetime')
plt.show()