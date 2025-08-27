import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os

# Set seaborn style
sns.set_style("whitegrid")

# Create output directory if it doesn't exist
output_dir = output_dir = "C:\\PythonProjects\\Simulations\\GPS"
os.makedirs(output_dir, exist_ok=True)

# Define spatial and temporal domains
x = np.linspace(-10, 10, 400)
t = np.linspace(0, 10, 100)
X, T_grid = np.meshgrid(x, t)

# Define symbolic fields
phi = np.exp(-((X - 2*np.sin(T_grid))**2)) * np.cos(T_grid)  # Fog density field
T_field = np.sin(X + T_grid) * np.exp(-0.1 * X**2)  # Torsion flux field

# Compute symbolic curvature gradients
curvature_phi = np.gradient(np.gradient(phi, axis=1), axis=1)
curvature_T = np.gradient(np.gradient(T_field, axis=1), axis=1)

# Coherence locking and entropy flow (symbolic representation)
coherence_locking = np.abs(curvature_phi - curvature_T)
entropy_flow = np.gradient(phi * T_field, axis=0)

# Create a time-lapse visualization
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap("plasma")

frames = []
for i in range(0, len(t), 5):
    ax.clear()
    sns.heatmap(coherence_locking[i].reshape(1, -1), cmap=cmap, cbar=True, ax=ax,
                xticklabels=50, yticklabels=False)
    ax.set_title(f"Coherence Locking at t={t[i]:.2f}", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.canvas.draw()
    frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    fig.savefig(frame_path)
    frames.append(frame_path)

# Save final frame as representative image
final_image_path = os.path.join(output_dir, "fog_torsion_timelapse.png")
fig.savefig(final_image_path)
