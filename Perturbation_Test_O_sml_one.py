import numpy as np
import matplotlib.pyplot as plt

# Simulate torsion-motivic measure under sigma perturbation
def torsion_motivic(sigma, alpha_SFIT):
    return np.exp(-alpha_SFIT * (1 - sigma)**2)

sigmas = np.linspace(0.6, 1.0, 100)
alpha_vals = [0.5, 1.0, 2.0]

for alpha in alpha_vals:
    plt.plot(sigmas, torsion_motivic(sigmas, alpha), label=f'α_SFIT={alpha}')

plt.xlabel('σ (Bifurcation Threshold)')
plt.ylabel('Torsion-Motivic Coherence')
plt.title('SFIT Torsion Stability under σ < 1')
plt.legend()
plt.grid(True)
plt.show()
