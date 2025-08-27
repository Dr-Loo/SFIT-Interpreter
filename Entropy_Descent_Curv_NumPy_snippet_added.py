import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Parameters
N = 50  # number of modes
T = 100  # time steps

# 1. Define target distribution p* (favors high-curvature modes)
curvatures = np.random.uniform(0, 1, N)
p_star = np.exp(3 * curvatures)  # Exponential preference for high curvature
p_star /= p_star.sum()  # Normalize

# 2. Initialize perturbed distribution p
p = p_star * np.random.uniform(0.5, 1.5, N)
p /= p.sum()

# 3. Define braid compatibility matrix (random positive definite)
W = np.random.uniform(0, 1, (N, N))
W = (W + W.T) / 2  # Symmetrize
np.fill_diagonal(W, 1.0)  # Self-compatibility = 1

# 4. Dynamics: Trade-off between curvature optimization and braid compatibility
entropy_trace = []
divergence_trace = []
alignment_trace = []
compatibility_trace = []

for t in range(T):
    # Gradient descent: minimizes D(p||p*) but constrained by braid compatibility
    gradient = np.log(p / p_star + 1e-12) + 1  # ∇D(p||p*)
    
    # Braid compatibility force: pulls toward compatible modes
    compatibility_force = W @ p / (p @ W @ p + 1e-12)
    
    # Combined update: balance between optimization and constraints
    update = -0.1 * gradient + 0.3 * compatibility_force
    
    # Update probabilities (projected gradient descent)
    p = p * np.exp(update)
    p /= p.sum()  # Renormalize
    
    # Track metrics
    entropy = -np.sum(p * np.log(p + 1e-12))
    divergence = np.sum(p * np.log(p / p_star + 1e-12))
    alignment = np.sum(p * curvatures)
    compatibility = p @ W @ p  # Overall compatibility score
    
    entropy_trace.append(entropy)
    divergence_trace.append(divergence)
    alignment_trace.append(alignment)
    compatibility_trace.append(compatibility)

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

ax1.plot(entropy_trace)
ax1.set_title('Entropy $S_U$ ↓')
ax1.set_ylabel('Entropy')

ax2.plot(divergence_trace)
ax2.set_title('Divergence $D(p||p^*)$ ↑')
ax2.set_ylabel('Divergence')

ax3.plot(alignment_trace)
ax3.set_title('Curvature Alignment ↓')
ax3.set_ylabel('Alignment')

ax4.plot(compatibility_trace)
ax4.set_title('Braid Compatibility ↑')
ax4.set_ylabel('Compatibility')

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final statistics
print("Final results:")
print(f"Entropy: {entropy_trace[0]:.3f} → {entropy_trace[-1]:.3f} (Δ = {entropy_trace[-1]-entropy_trace[0]:.3f})")
print(f"Divergence: {divergence_trace[0]:.3f} → {divergence_trace[-1]:.3f} (Δ = {divergence_trace[-1]-divergence_trace[0]:.3f})")
print(f"Alignment: {alignment_trace[0]:.3f} → {alignment_trace[-1]:.3f} (Δ = {alignment_trace[-1]-alignment_trace[0]:.3f})")
print(f"Compatibility: {compatibility_trace[0]:.3f} → {compatibility_trace[-1]:.3f} (Δ = {compatibility_trace[-1]-compatibility_trace[0]:.3f})")