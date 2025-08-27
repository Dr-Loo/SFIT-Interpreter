import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import zeta
from scipy.fft import fft
import pandas as pd
import os

# Set seaborn style
sns.set(style="whitegrid")

# Create output directory
output_dir = "/mnt/data"
os.makedirs(output_dir, exist_ok=True)

# Simulate moduli lattice scan
moduli_size = 500
moduli_lattice = np.linspace(1.01, 50, moduli_size)

# Compute zeta function values
zeta_values = zeta(2, moduli_lattice)

# Estimate residues of higher-order poles (numerical derivative)
zeta_derivative = np.gradient(zeta_values, moduli_lattice)
residues = zeta_derivative / zeta_values

# Fractal braid density analysis via FFT
braid_density = np.abs(fft(residues))

# Entropy flux correlation (Shannon entropy of braid density)
def shannon_entropy(data):
    prob = data / np.sum(data)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

entropy_flux = shannon_entropy(braid_density)

# Create DataFrame for analysis
results_df = pd.DataFrame({
    'Moduli': moduli_lattice,
    'Zeta': zeta_values,
    'Residue': residues,
    'BraidDensity': braid_density
})

# Save results to CSV
csv_path = os.path.join(output_dir, "moduli_zeta_analysis.csv")
results_df.to_csv(csv_path, index=False)

# Plot residues
plt.figure(figsize=(10, 6))
sns.lineplot(x=moduli_lattice, y=residues, palette="viridis")
plt.title("Residues of Higher-Order Zeta Poles")
plt.xlabel("Moduli")
plt.ylabel("Residue")
residue_plot_path = os.path.join(output_dir, "residue_plot.png")
plt.savefig(residue_plot_path)
plt.close()

# Plot braid density
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(len(braid_density)), y=braid_density, palette="plasma")
plt.title("Fractal Braid Density Spectrum")
plt.xlabel("Frequency Index")
plt.ylabel("Density")
braid_plot_path = os.path.join(output_dir, "braid_density_plot.png")
plt.savefig(braid_plot_path)
plt.close()

# Print entropy flux
print(f"Entropy Flux: {entropy_flux:.4f}")
