import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Torsion field
def torsion_field(x, y):
    return np.sin(x) * np.cos(y)

# Dirac operator (simplified)
def dirac_operator(x, y, torsion):
    gamma0 = np.array([[0, 1], [1, 0]])
    gamma1 = np.array([[0, -1j], [1j, 0]])
    gamma2 = np.array([[1, 0], [0, -1]])
    H = gamma0 + gamma1 * x + gamma2 * y + torsion * gamma0
    return H

# Simulate wavefunction magnitude
wavefunction_magnitude = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        torsion = torsion_field(X[i, j], Y[i, j])
        H = dirac_operator(X[i, j], Y[i, j], torsion)
        psi = expm(-1j * H)[:, 0]
        wavefunction_magnitude[i, j] = np.abs(psi[0])**2 + np.abs(psi[1])**2

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, wavefunction_magnitude, levels=100, cmap='plasma')
plt.colorbar(label='Wavefunction Magnitude')
plt.title('Fermion Wavefunction Profiles Across Torsion Basins')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
