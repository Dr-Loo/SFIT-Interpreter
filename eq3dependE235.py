import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Set seaborn style for aesthetics
sns.set(style="whitegrid")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define positions of exceptional fibers in Σ(2,3,5)
exceptional_fibers = {
    '2': (2, 5),
    '3': (4, 3),
    '5': (6, 7)
}

# Plot exceptional fibers
for label, (x, y) in exceptional_fibers.items():
    ax.plot(x, y, 'o', markersize=10, label=f'Fiber {label}')
    ax.text(x + 0.2, y + 0.2, f'ρ_D({label})', fontsize=12)

# Simulate Braid β₃ (trefoil) action as a loop connecting fibers
braid_path = np.array([
    [2, 5], [3, 6], [4, 3], [5, 4], [6, 7]
])
ax.plot(braid_path[:, 0], braid_path[:, 1], linestyle='--', color='purple', linewidth=2, label='Braid β₃ action')

# Highlight torsion flux as arrows between fibers
arrow_style = dict(arrowstyle="->", color="red", linewidth=2)
arrow1 = FancyArrowPatch((2, 5), (4, 3), **arrow_style)
arrow2 = FancyArrowPatch((4, 3), (6, 7), **arrow_style)
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# Motivic stratification: shaded regions
ax.fill_between([1.5, 4.5], 2, 6, color='lightblue', alpha=0.3, label='Motivic Layer 1')
ax.fill_between([4.5, 6.5], 4, 8, color='lightgreen', alpha=0.3, label='Motivic Layer 2')

# Final plot adjustments
ax.set_title("Visualization of Braid β₃ Acting on E₃^ℚ ⊂ Σ(2,3,5)", fontsize=16)
ax.set_xlabel("X-axis (abstract fiber space)", fontsize=12)
ax.set_ylabel("Y-axis (localization coordinates)", fontsize=12)
ax.legend()
ax.set_xlim(1, 7)
ax.set_ylim(1, 9)

plt.show()
