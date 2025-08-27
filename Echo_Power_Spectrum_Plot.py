import numpy as np
import matplotlib.pyplot as plt

# Parameters
m_H = 125.0            # Higgs mass (GeV)
Lambda_torsion = 2.49  # Reconnection scale (GeV)
beta = 0.01            # Damping factor
k_vals = np.linspace(0.1, 10, 500)  # Wave numbers

# Correlation function (radial decay)
def scalar_correlation(r):
    return np.log(1 + m_H) * np.exp(-r / (1.0 / Lambda_torsion))

# Echo power spectrum integrand
def P_echo_integrand(k):
    r_vals = np.linspace(0.01, 10, 1000)
    integrand = scalar_correlation(r_vals) * np.sin(k * r_vals) * r_vals**2
    return np.trapz(integrand, r_vals)

# Compute power spectrum
P_echo = np.array([P_echo_integrand(k)**2 * np.exp(-beta * (k**2)) for k in k_vals])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_vals, P_echo, color='mediumorchid', linewidth=2)
plt.title('Scalar Fog Echo Power Spectrum', fontsize=14)
plt.xlabel('Wave Number k', fontsize=12)
plt.ylabel('Power P_echo(k)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
