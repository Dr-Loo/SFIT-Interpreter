import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


L = 100  # grid size
T = 100  # time steps
dx = 1.0
dt = 0.1
hbar = 1.0
m = 1.0

psi = np.zeros((L, L), dtype=np.complex128)
psi[L//2, L//2] = 1.0 + 0j

torsion = np.random.normal(0, 0.1, (L, L))

def laplacian(field):
    return (-4 * field + np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)) / dx**2

for t in range(T):
    lap = laplacian(psi)
    torsion_effect = torsion * psi
    dpsi_dt = (-1j * hbar / (2 * m)) * lap + 1j * torsion_effect
    psi += dpsi_dt * dt
    torsion += np.random.normal(0, 0.01, (L, L))

prob_density = np.abs(psi)**2

plt.figure(figsize=(8, 6))
sns.heatmap(prob_density, cmap="plasma", cbar_kws={'label': 'Probability Density'})
plt.title("2D Fermion Wavefunction in Dynamic Torsion Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
