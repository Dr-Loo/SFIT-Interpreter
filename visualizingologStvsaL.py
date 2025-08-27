import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set seaborn style
sns.set(style="whitegrid")

# Create output directory
output_dir = "/mnt/data"
os.makedirs(output_dir, exist_ok=True)

# Simulate synthetic SFIT prediction and observed data
np.random.seed(42)
n_samples = 500

# Parameters
alpha = np.random.uniform(0.1, 2.0, n_samples)
chi_orb = np.random.uniform(0.01, 1.0, n_samples)
logS_t = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

# Simulate standard deviation of logS(t)
sigma_logS_t = np.abs(np.random.normal(loc=0.5, scale=0.2, size=n_samples))

# Simulate bifurcation intensity zones (categorical)
bifurcation_zone = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)

# Simulate curvature well depth clusters (categorical)
curvature_cluster = np.random.choice(['Shallow', 'Moderate', 'Deep'], size=n_samples)

# Create DataFrame
data = pd.DataFrame({
    'alpha': alpha,
    '|chi_orb|': chi_orb,
    'σ(logS(t))': sigma_logS_t,
    'Bifurcation Zone': bifurcation_zone,
    'Curvature Cluster': curvature_cluster
})

# Create heatmap grid
heatmap_data = data.pivot_table(index='alpha', columns='|chi_orb|', values='σ(logS(t))', aggfunc='mean')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='plasma', cbar_kws={'label': 'σ(logS(t))'})
plt.title('Heatmap of σ(logS(t)) vs. (α, |χorb|)')
plt.xlabel('|χorb|')
plt.ylabel('α')

# Save plot
heatmap_path = os.path.join(output_dir, "heatmap_sigma_logS_vs_alpha_chiorb.png")
plt.savefig(heatmap_path)
plt.close()
