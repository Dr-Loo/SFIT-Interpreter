import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label

def compute_symbolic_curvature(moduli_lattice, residue_map, alpha=0.7):
    """
    Compute symbolic curvature as entropy-weighted residue gradient.
    moduli_lattice: 2D numpy array of symbolic moduli values
    residue_map: 2D numpy array of residue field
    alpha: mixing factor between moduli gradient and residue entropy
    """
    grad_moduli_x, grad_moduli_y = np.gradient(moduli_lattice)
    grad_residue_x, grad_residue_y = np.gradient(residue_map)

    entropy_flux = np.log1p(np.abs(residue_map))  # symbolic entropy proxy
    curvature_raw = alpha * (grad_moduli_x**2 + grad_moduli_y**2) + \
                    (1 - alpha) * entropy_flux * (grad_residue_x**2 + grad_residue_y**2)

    curvature_field = gaussian_filter(curvature_raw, sigma=1.5)
    return curvature_field

def detect_curvature_wells(curvature_field, threshold_ratio=0.85):
    """
    Detect curvature wells by thresholding high-torsion regions.
    """
    threshold = threshold_ratio * np.max(curvature_field)
    wells_mask = curvature_field > threshold

    labeled_wells, num_wells = label(wells_mask)
    return labeled_wells, num_wells

def plot_curvature_field(curvature_field, labeled_wells=None):
    plt.figure(figsize=(12,6))
    sns.heatmap(curvature_field, cmap="magma")
    plt.title("Symbolic Curvature Wells")
    plt.xlabel("Moduli X-axis")
    plt.ylabel("Moduli Y-axis")
    if labeled_wells is not None:
        plt.contour(labeled_wells, colors="cyan", linewidths=1)
    plt.show()

# --- Usage Example ---
if __name__ == "__main__":
    # Replace this with actual lattice and residue inputs from SFIT simulation
    moduli_lattice = np.random.randn(100, 100)
    residue_map = np.sin(moduli_lattice) * np.random.randn(100, 100)

    curvature_field = compute_symbolic_curvature(moduli_lattice, residue_map)
    wells, count = detect_curvature_wells(curvature_field)
    print(f"Detected {count} symbolic curvature wells.")
    plot_curvature_field(curvature_field, labeled_wells=wells)
