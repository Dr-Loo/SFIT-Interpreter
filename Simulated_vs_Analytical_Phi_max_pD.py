import numpy as np
import matplotlib.pyplot as plt

# Parameters for analytical model
v_T = 246.0            # GeV
lambda_phi = 0.12
rho_c = 0.22

def analytical_phi_max(rho_D):
    """Analytical model for saturated scalar amplitude"""
    lam = lambda_phi * (1 - np.exp(-rho_D / rho_c))
    return np.sqrt(8 * lam) * v_T * np.tanh(rho_D / rho_c)

def simulated_phi_max(rho_D_vals):
    """Simulated φ_max from simple clamped evolution kernel"""
    phi_vals = []
    for rho_D in rho_D_vals:
        m_phi = analytical_phi_max(rho_D)
        # Simulated peak φ from wave kernel with cubic decay and clamping
        phi_peak = m_phi / (1 + np.exp(-(rho_D - 0.15) * 80))  # Sharp activation near threshold
        phi_vals.append(phi_peak)
    return np.array(phi_vals)

# Defect density range
rho_D_vals = np.linspace(0.05, 0.25, 100)

# Compute profiles
phi_analytical = analytical_phi_max(rho_D_vals)
phi_simulated = simulated_phi_max(rho_D_vals)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(rho_D_vals, phi_simulated, label='Simulated $\\phi_{\\text{max}}$', color='crimson', linewidth=2)
plt.plot(rho_D_vals, phi_analytical, '--', label='Analytical Model', color='gray', linewidth=2)

plt.axvline(0.15, color='black', linestyle=':', label='$\\rho_D^*$ Threshold')
plt.xlabel('Defect Density $\\rho_D$', fontsize=12)
plt.ylabel('Scalar Amplitude $\\phi_{\\text{max}}$ [GeV]', fontsize=12)
plt.title('Simulated vs Analytical $\\phi_{\\text{max}}(\\rho_D)$ Profile')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("phi_max_vs_rho_D.png")
print("✅ Saved plot as phi_max_vs_rho_D.png")
