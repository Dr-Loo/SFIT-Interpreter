import numpy as np

# Define identifiers
identifiers = np.array(['m1839', 'm1840', 'm1841', 'm1842', 'm1843', 'm1844', 'm1845', 'm1846', 'm1847'])

# Baseline scalar field amplitude
baseline = 6e-05

# Pulse parameters
pulse_center_index = 0  # corresponds to m1839
pulse_strength = 1.1 * baseline
sigma = 1.5  # controls gradient spread

# Generate entropy gradient pulse using Gaussian
pulse = np.array([
    pulse_strength * np.exp(-((i - pulse_center_index) ** 2) / (2 * sigma ** 2))
    for i in range(len(identifiers))
])

# Track output: updated scalar amplitudes
for ident, amp in zip(identifiers, pulse):
    print(f"{ident}\t{amp:.8f}")
