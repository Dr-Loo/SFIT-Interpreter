import numpy as np
import matplotlib.pyplot as plt

# Critical defect density
rho_c = 0.15
rho_D = np.linspace(0, 0.3, 500)

# SFIT-style binary locking activation (not used for smooth curve, but included)
def activation(rho):
    return np.where(rho >= rho_c, 1.0, 0.0)

# Smooth locking curve (sigmoid transition near rho_c)
def locking_prob(rho):
    return 1 / (1 + np.exp(-50 * (rho - rho_c)))

# Torsion gradient modifier — Gaussian bump around rho_c
def torsion_mod(rho, amp=0.35, width=0.025):
    return 1 + amp * np.exp(-((np.clip(rho, 0, rho_c) - rho_c)**2) / (2 * width**2))

# Compute base and torsion-modulated locking curves
base_curve = locking_prob(rho_D)
torsion_curve = base_curve * torsion_mod(rho_D)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(rho_D, base_curve, label='No Torsion Feedback', linestyle='--', color='steelblue')
plt.plot(rho_D, torsion_curve, label='Torsion-Mediated Locking', color='darkorange')
plt.axvline(rho_c, color='crimson', linestyle=':', label=r'Critical Density $\rho_c = 0.15$')
plt.axvspan(0, rho_c, alpha=0.1, color='gray', label='Fog Phase')

# Axis labels with SFIT terminology
plt.xlabel('Defect Density $\\rho_D$ (GeV$^3$)')
plt.ylabel('Scalar Field Amplitude $\\phi/\\phi_{\\rm max}$')
plt.title('SFIT Fog-to-Locking Transition with Torsion Gradient')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
