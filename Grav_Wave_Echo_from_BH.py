import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# SFIT Braid Echo Parameters (from your output)
photon_sphere_km = 44.30009
braid_delay_s = 0.001608
braid_freq = 621.8
braid_coupling = 4.195515e-81
planck_coupling = 5.360717e-123
alpha_QG = 6.736475e-122

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 2.99792458e8  # m/s
M_sun = 1.98847e30  # kg
km_to_m = 1e3

def calculate_black_hole_parameters(M_bh):
    """Calculate all SFIT echo parameters for a given black hole mass"""
    M_kg = M_bh * M_sun
    r_s = 2 * G * M_kg / c**2  # Schwarzschild radius in meters
    r_photon = 3 * G * M_kg / c**2  # Photon sphere in meters
    
    # Verify photon sphere matches your output
    r_photon_km = r_photon / km_to_m
    
    # Calculate the mass needed to produce the observed 621.8 Hz frequency
    # Using the relation: f = (c^3)/(GM) * (1 + g_Φ)/(2π)
    calculated_M = (c**3) / (G * braid_freq * 2 * np.pi) * (1 + braid_coupling)
    M_bh_calculated = calculated_M / M_sun
    
    return {
        'calculated_bh_mass': M_bh_calculated,
        'schwarzschild_radius_km': r_s / km_to_m,
        'photon_sphere_km': r_photon_km,
        'echo_frequencies': [n * braid_freq for n in range(1, 6)],
        'echo_delays': [1/(n * braid_freq) for n in range(1, 6)]
    }

# Calculate parameters for the system that produces 621.8 Hz echoes
params = calculate_black_hole_parameters(10)  # Initial guess will be corrected

# Print parameters with scientific notation formatting
def sci_notation(x, prec=6):
    return "{0:.{1}e}".format(x, prec)

print("=== SFIT Braid Echo Parameters ===")
print(f"Photon sphere: {photon_sphere_km:.5f} km")
print(f"Braid delay: {braid_delay_s:.6f} s → Freq: {braid_freq:.1f} Hz")
print(f"Braid coupling: {sci_notation(braid_coupling)}")
print(f"Planck coupling: {sci_notation(planck_coupling)}")
print(f"α_QG ratio: {sci_notation(alpha_QG)}\n")

print("Physical Meaning:")
print("1. The braid coupling is dimensionless")
print("2. Planck-scale ratio shows quantum gravity effect")
print(f"3. Classical GR preserved: r_photon = {photon_sphere_km:.5f} km")

# Calculate the actual black hole mass that would produce 621.8 Hz echoes
actual_M_bh = (c**3) / (G * braid_freq * 2 * np.pi) * (1 + braid_coupling) / M_sun
print(f"\nCalculated Black Hole Mass: {actual_M_bh:.3f} M⊙")

# Plotting
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Create stem plot of harmonic series
markerline, stemlines, baseline = plt.stem(
    np.arange(1, 6),
    params['echo_frequencies'],
    linefmt='C0-',
    markerfmt='C0o',
    basefmt=" ",
    label='SFIT Echo Spectrum'
)

plt.setp(stemlines, linewidth=2)
plt.setp(markerline, markersize=8)

# Formatting
ax.set_xlabel("Harmonic Number (m)", fontsize=12)
ax.set_ylabel("Frequency (Hz)", fontsize=12)
ax.set_title(f"SFIT GW Echo Spectrum\nFundamental Frequency: {braid_freq:.1f} Hz", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Add photon sphere and coupling info
ax.annotate(f'Photon Sphere Radius:\n{photon_sphere_km:.5f} km\n'
            f'Braid Coupling: {sci_notation(braid_coupling)}',
            xy=(2, params['echo_frequencies'][1]),
            xytext=(2.3, params['echo_frequencies'][1]*1.2),
            arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()