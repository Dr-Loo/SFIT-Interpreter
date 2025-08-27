import numpy as np
import matplotlib.pyplot as plt

# Parameters for a 10M⊙ black hole
M = 10 * 1.989e30  # kg
G = 6.67430e-11    # m^3 kg^-1 s^-2
c = 3e8            # m/s
hbar = 1.0545718e-34  # J·s
g_phi = 1e-12      # SFIT coupling constant (adjustable)

def omega_m(m, g_phi):
    return (m * c**3 / (G * M)) * (1 + (hbar * g_phi) / (4 * np.pi))

# Predicted frequencies for m=1,2,3
frequencies = [omega_m(m, g_phi) / (2 * np.pi) for m in [1, 2, 3]]
sub_harmonics = [omega_m(m, g_phi) / (4 * np.pi) for m in [1, 2, 3]]

print(f"Fundamental frequencies (Hz): {frequencies}")
print(f"Sub-harmonics (Hz): {sub_harmonics}")

# Plot idealized echo spectrum
plt.stem(frequencies, np.ones_like(frequencies), linefmt='b-', label='ω_m')
plt.stem(sub_harmonics, 0.5 * np.ones_like(sub_harmonics), linefmt='r--', label='ω_m/2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (arb.)')
plt.legend()
plt.title('SFIT-XSM Predicted GW Echo Spectrum')
plt.show()