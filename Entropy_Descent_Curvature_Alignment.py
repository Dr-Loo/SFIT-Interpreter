# SFIT-XSM Simulation: Entropy Descent & Curvature Alignment

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100  # number of symbolic modes
T = 200  # time steps
fusion_rate = 0.1
fission_rate = 0.05
curvature_bias = 0.3  # strength of holonomy preference

# Initialize modes with curvature (holonomy measure)
# High curvature = better holonomy preservation
modes = [{'label': f'M{j}', 'curvature': np.random.uniform(0, 1)} for j in range(N)]

# Target distribution p* that favors high-curvature (holonomic) modes
# p* ∝ exp(β * curvature) where β controls the strength of preference
beta = 3.0
p_star = np.array([np.exp(beta * m['curvature']) for m in modes])
p_star = p_star / np.sum(p_star)  # Normalize

# Initial probability distribution (perturbed from target)
p = p_star * np.random.uniform(0.5, 1.5, N)
p = p / np.sum(p)

# Trackers
entropy_trace = []
relative_entropy_trace = []  # D(p||p*)
curvature_alignment_trace = []
fusion_events = []
fission_events = []

# Dynamics Loop with Detailed Balance
for t in range(T):
    t_fusion_events = 0
    t_fission_events = 0
    
    # Fusion process: B_α ⊗ B_β → B_γ
    for _ in range(int(fusion_rate * N)):
        i, j = np.random.choice(N, 2, replace=False)
        # Prefer fusion of modes with similar curvature (holonomy class)
        if abs(modes[i]['curvature'] - modes[j]['curvature']) < curvature_bias:
            # Choose target mode γ with probability proportional to p*_γ
            k = np.random.choice(N, p=p_star)
            
            # Detailed balance condition: W_{ij→k} p*_i p*_j = W_{k→ij} p*_k
            # We implement this by accepting the fusion with probability:
            accept_prob = min(1, (p_star[k] * fission_rate) / 
                             (p_star[i] * p_star[j] * fusion_rate + 1e-12))
            
            if np.random.random() < accept_prob:
                p[k] += p[i] + p[j]
                p[i] = 0
                p[j] = 0
                t_fusion_events += 1

    # Fission process: B_γ → B_α ⊗ B_β
    for _ in range(int(fission_rate * N)):
        k = np.random.choice(N)
        if p[k] > 0.01:  # Only split modes with significant probability
            i, j = np.random.choice(N, 2, replace=False)
            
            # Detailed balance condition
            accept_prob = min(1, (p_star[i] * p_star[j] * fusion_rate) / 
                             (p_star[k] * fission_rate + 1e-12))
            
            if np.random.random() < accept_prob:
                split_amount = p[k] * 0.5
                p[i] += split_amount
                p[j] += split_amount
                p[k] -= split_amount
                t_fission_events += 1

    # Normalize probabilities
    p = p / np.sum(p)
    
    # Track metrics
    entropy = -np.sum(p * np.log(p + 1e-12))
    relative_entropy = np.sum(p * np.log(p / (p_star + 1e-12) + 1e-12))
    curvature_alignment = np.sum(p * np.array([m['curvature'] for m in modes]))
    
    entropy_trace.append(entropy)
    relative_entropy_trace.append(relative_entropy)
    curvature_alignment_trace.append(curvature_alignment)
    fusion_events.append(t_fusion_events)
    fission_events.append(t_fission_events)

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Entropy plot
ax1.plot(entropy_trace)
ax1.set_title('Symbolic Entropy $S_U(t)$')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy')
ax1.grid(True, alpha=0.3)

# Relative entropy plot (Lyapunov function)
ax2.plot(relative_entropy_trace)
ax2.set_title('Relative Entropy $D(p||p^*)$')
ax2.set_xlabel('Time')
ax2.set_ylabel('$D(p||p^*)$')
ax2.grid(True, alpha=0.3)

# Curvature alignment plot
ax3.plot(curvature_alignment_trace)
ax3.set_title('Curvature Alignment')
ax3.set_xlabel('Time')
ax3.set_ylabel('Weighted Curvature')
ax3.grid(True, alpha=0.3)

# Event rates
ax4.plot(fusion_events, label='Fusion events')
ax4.plot(fission_events, label='Fission events')
ax4.set_title('Fusion/Fission Events')
ax4.set_xlabel('Time')
ax4.set_ylabel('Events per timestep')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final statistics
print(f"Initial entropy: {entropy_trace[0]:.4f}")
print(f"Final entropy: {entropy_trace[-1]:.4f}")
print(f"Entropy change: {entropy_trace[-1] - entropy_trace[0]:.4f}")
print(f"Initial relative entropy: {relative_entropy_trace[0]:.4f}")
print(f"Final relative entropy: {relative_entropy_trace[-1]:.4f}")
print(f"Initial curvature alignment: {curvature_alignment_trace[0]:.4f}")
print(f"Final curvature alignment: {curvature_alignment_trace[-1]:.4f}")