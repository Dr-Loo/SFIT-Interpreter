import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude, median_filter
from sklearn.cluster import DBSCAN

# ===== IMPROVED COLOR MAPPING =====
def rgb_to_value(rgb_row):
    r, g, b = rgb_row["R"], rgb_row["G"], rgb_row["B"]
    
    # Relaxed color ranges (adjust as needed)
    if r > 180 and g > 180 and b < 80:  # Yellow
        return 1.91 + np.random.normal(0, 0.01)  # Add small noise
    elif g > 80 and r < 120 and b < 120:  # Green
        return 1.88 + np.random.normal(0, 0.01)
    elif b > 150 and r < 80 and g < 80:  # Blue
        return 1.85 + np.random.normal(0, 0.01)
    elif r > 80 and b > 150 and g < 80:  # Violet
        return 1.84 + np.random.normal(0, 0.01)
    else:
        return np.nan

# ===== IMPROVED PEAK DETECTION =====
def find_peaks(df):
    """Identify true peaks using prominence."""
    from scipy.signal import find_peaks
    
    # Sample every 10th point for speed (adjust as needed)
    sample = df.iloc[::10].copy()
    peaks, _ = find_peaks(sample["Value"], 
                         prominence=0.02,  # Min height difference from surroundings
                         distance=20)      # Min spacing between peaks
    return sample.iloc[peaks]

# ===== IMPROVED EDGE DETECTION =====
def find_edges(grid):
    """Use adaptive thresholding."""
    gradient = gaussian_gradient_magnitude(grid, sigma=2)
    
    # Otsu's automatic thresholding
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(gradient[~np.isnan(gradient)])
        return gradient > (0.5 * thresh)  # Conservative threshold
    except:
        return np.zeros_like(grid, dtype=bool)

# ===== MAIN ANALYSIS =====
def analyze_data(df):
    # Create grid with interpolation
    grid = df.pivot(index="y", columns="x", values="Value").values
    grid_filled = median_filter(grid, size=5)
    
    # Detect features
    peaks = find_peaks(df)
    edge_mask = find_edges(grid_filled)
    edges = np.argwhere(edge_mask)
    
    # Clustering with automatic eps tuning
    coords = df[["x", "y", "Value"]].values
    clustering = DBSCAN(eps=7, min_samples=15).fit(coords)  # Increased params
    df["Domain"] = clustering.labels_
    
    return peaks, edges, df[df["Domain"] != -1], grid_filled

# ===== RUN ANALYSIS =====
if __name__ == "__main__":
    df = pd.read_csv("domain_data.csv")
    df["Value"] = df.apply(rgb_to_value, axis=1)
    df = df.dropna().reset_index(drop=True)
    
    peaks, edges, domains, grid = analyze_data(df)
    
    # Save results
    peaks.to_csv("peaks.csv", index=False)
    pd.DataFrame(edges, columns=["y", "x"]).to_csv("edges.csv", index=False)
    domains.to_csv("domains.csv", index=False)
    
    # Enhanced visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(grid, cmap="viridis", origin="lower")
    plt.colorbar(im, label="Value")
    ax.scatter(peaks["x"], peaks["y"], c="red", s=10, label=f"Peaks ({len(peaks)})")
    ax.scatter(edges[:,1], edges[:,0], c="black", s=1, alpha=0.3, 
              label=f"Edges ({len(edges)})")
    ax.legend()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    plt.close()