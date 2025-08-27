import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# Set seaborn style
sns.set(style="whitegrid")

# Simulation parameters
np.random.seed(42)
time_steps = 1000
control_param = 0.05  # Control parameter for torsion-entropy coupling
entropy_base = 1.0

# Initialize arrays
torsion = np.zeros(time_steps)
entropy = np.zeros(time_steps)
phase = np.zeros(time_steps)

# Initial conditions
torsion[0] = 0.1
entropy[0] = entropy_base
phase[0] = 0.0

# Simulate torsion-entropy interaction with quantum analog mappings
for t in range(1, time_steps):
    # Motivic drift: entropy fluctuates with torsion feedback
    entropy[t] = entropy[t-1] + control_param * np.sin(torsion[t-1]) + np.random.normal(0, 0.01)
    
    # Torsion evolves with entropy influence and phase locking
    torsion[t] = torsion[t-1] + control_param * np.cos(entropy[t-1]) + np.random.normal(0, 0.01)
    
    # Phase evolution with symmetry breaking and scars
    phase[t] = phase[t-1] + control_param * np.sin(torsion[t]) * np.cos(entropy[t])

# Detect resonant locking via peak analysis
peaks, _ = find_peaks(np.abs(np.diff(phase)), height=0.05)

# Save results to CSV
data = pd.DataFrame({
    'Time': np.arange(time_steps),
    'Torsion': torsion,
    'Entropy': entropy,
    'Phase': phase
})
save_path_csv = "C:\\PythonProjects\\Simulations\\SFIT-XSM\\torsion_entropy_simulation_results.csv"
save_path_png = "C:\\PythonProjects\\Simulations\\SFIT-XSM\\torsion_entropy_phase_plot.png"

data.to_csv(save_path_csv, index=False)
plt.savefig(save_path_png)


# Plot phase evolution with scars and symmetry breaking
plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['Phase'], label='Phase')
plt.scatter(peaks, phase[peaks], color='red', label='Resonant Locks')
plt.title('Phase Evolution with Resonant Locking and Symmetry Breaking')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/torsion_entropy_phase_plot.png")
plt.close()

print("Simulation completed. Results saved.")
import os
print("CSV saved?", os.path.exists(save_path_csv))
print("Plot saved?", os.path.exists(save_path_png))
