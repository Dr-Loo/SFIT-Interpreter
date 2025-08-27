import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from matplotlib.animation import FuncAnimation

# ========================
# 1. SETUP DEFECT LATTICE
# ========================
N = 128
rho_D = np.zeros((N, N))

# Central black hole defect (elliptical)
rho_D[N//2-10:N//2+10, N//2-5:N//2+15] = 1.0  

# Off-center instanton
rho_D[N//3:N//3+20, 2*N//3:2*N//3+20] = 0.7  

# Quantum foam
rho_D += 0.05 * np.random.rand(N, N)  

# ========================
# 2. TOPOLOGICAL SOURCES
# ========================
def black_hole_term(r, r_s=15):
    """Gaussian horizon profile"""
    return np.exp(-(r - r_s)**2 / 2)

r = np.linalg.norm(np.indices((N, N)) - N//2, axis=0)
H_BH = black_hole_term(r)  # BH horizon contribution

# ========================
# 3. EMERGENCE SIMULATION
# ========================
phi = np.zeros_like(rho_D)
steps = 100
history = []  # Store phi at each step

for _ in range(steps):
    phi += 0.01 * (laplace(phi, mode='wrap') + rho_D + 0.5*H_BH)
    history.append(phi.copy())

# ========================
# 4. NUMERICAL ANALYSIS
# ========================
max_phi = np.max(phi)
min_phi = np.min(phi)
bh_peak = phi[N//2, N//2]  # Value at BH center
instanton_peak = phi[N//3, 2*N//3]  # Value at instanton

print(f"""
=== NUMERICAL RESULTS ===
1. Spacetime Field (ϕ) Range: 
   - Max: {max_phi:.3f} (strongest gravity)
   - Min: {min_phi:.3f} (flattest spacetime)
   
2. Topological Peaks:
   - BH Core ϕ: {bh_peak:.3f}
   - Instanton ϕ: {instanton_peak:.3f}
   
3. Defect Statistics:
   - Total Defect Mass: {np.sum(rho_D):.1f} 
   - BH Horizon Strength: {np.sum(H_BH):.1f}
""")

# ========================
# 5. ANIMATION
# ========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(frame):
    ax1.clear()
    ax2.clear()
    
    # Plot defect density
    ax1.imshow(rho_D + 0.5*H_BH, cmap='plasma')
    ax1.set_title(f'Defects + BH Horizon (Step {frame})')
    
    # Plot emergent spacetime
    ax2.imshow(history[frame], cmap='viridis')
    ax2.set_title(f'Emergent ϕ (Max: {np.max(history[frame]):.2f})')
    
    plt.tight_layout()

ani = FuncAnimation(fig, update, frames=steps, interval=100)
plt.show()