import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define functions with explicit dependencies
def P_echo(k, Lambda_torsion=1.0):
    """Power spectrum of scalar echoes with torsion cutoff."""
    return np.exp(-(k * Lambda_torsion)**2)

def sigma_echo_to_nu_nu(k, y_phi=1e-5, m_phi=70):
    """Cross-section for echoes → neutrinos."""
    return (y_phi**2 / m_phi**2) * (k**2 / (k**2 + m_phi**2))

def dNeff_dt(k_max=10.0, Lambda_torsion=1.0, y_phi=1e-5, m_phi=70):
    """Thermalization rate of echoes into relativistic species."""
    integrand = lambda k: P_echo(k, Lambda_torsion) * sigma_echo_to_nu_nu(k, y_phi, m_phi)
    return quad(integrand, 0, k_max)[0]

# Example usage
print(f"dN_eff/dt ~ {dNeff_dt():.2e}")  # Compare to Planck bounds (ΔN_eff < 0.25)