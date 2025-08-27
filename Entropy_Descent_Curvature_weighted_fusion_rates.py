# SFIT-XSM Simulation with Braid-Theoretic Constraints
# Implements curvature-weighted fusion and proper detailed balance

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

# Parameters
N = 100  # number of symbolic modes
T = 200  # time steps
base_fusion_rate = 0.08
base_fission_rate = 0.04
curvature_bias = 0.3  # strength of holonomy preference
braid_compatibility_strength = 2.0  # strength of braid constraints

# Initialize modes with curvature and braid properties
modes = []
for j in range(N):
    # Each mode has curvature and a "braid charge" (simplified)
    curvature = np.random.uniform(0, 1)
    braid_charge = np.random.choice([-1, 0, 1])  # Simplified braid property
    modes.append({
        'label': f'M{j}',
        'curvature': curvature,
        'braid_charge': braid_charge,
        'twist': np.random.uniform(0, 2*np.pi)  # Additional topological property
    })

# Target distribution p* favors high-curvature, holonomy-preserving modes
beta = 3.0
p_star = np.array([np.exp(beta * m['curvature']) for m in modes])
p_star = p_star / np.sum(p_star)

# Initial probability distribution
p = p_star * np.random.uniform(0.5, 1.5, N)
p = p / np.sum(p)

def compute_braid_compatibility(mode_i, mode_j, mode_k):
    """
    Compute Φ_{αβγ} based on braid compatibility and curvature alignment
    Implements the constraint factor for detailed balance
    """
    # 1. Curvature compatibility (holonomy preservation)
    curv_compat = np.exp(-(
        abs(mode_i['curvature'] - mode_j['curvature']) +
        abs(mode_i['curvature'] - mode_k['curvature']) +
        abs(mode_j['curvature'] - mode_k['curvature'])
    ) / curvature_bias)
    
    # 2. Braid charge conservation (simplified)
    # In proper fusion categories, charges must sum appropriately
    charge_sum = mode_i['braid_charge'] + mode_j['braid_charge']
    charge_ok = abs(charge_sum - mode_k['braid_charge']) < 0.5  # Tolerance
    charge_factor = 1.0 if charge_ok else 0.1  # Strong penalty for violation
    
    # 3. Twist compatibility (topological phase matching)
    twist_diff = abs(mode_i['twist'] + mode_j['twist'] - mode_k['twist'])
    twist_diff = min(twist_diff, 2*np.pi - twist_diff)  # Handle periodicity
    twist_factor = np.exp(-twist_diff / (np.pi/4))
    
    # 4. Fusion probability (simulating N_{αβ}^γ)
    # In real fusion categories, this would be 0 or 1 for forbidden/allowed
    # Here we use a probabilistic approach based on compatibility
    fusion_prob = curv_compat * charge_factor * twist_factor
    
    return fusion_prob

def compute_reverse_braid_compatibility(mode_k, mode_i, mode_j):
    """
    Compute Φ_{γαβ}^{-1} for the reverse process
    """
    # For detailed balance, we need the inverse compatibility
    # In many cases, this is just 1/Φ, but we need to handle zeros carefully
    forward_compat = compute_braid_compatibility(mode_i, mode_j, mode_k)
    return 1.0 / (forward_compat + 1e-12)  # Avoid division by zero

# Trackers
entropy_trace = []
relative_entropy_trace = []
curvature_alignment_trace = []
fusion_events = []
fission_events = []
compatibility_trace = []  # Track average braid compatibility

# Dynamics Loop with Braid-Theoretic Detailed Balance
for t in range(T):
    t_fusion_events = 0
    t_fission_events = 0
    total_compatibility = 0
    compatibility_count = 0
    
    # Fusion process: B_α ⊗ B_β → B_γ with braid constraints
    for _ in range(int(base_fusion_rate * N)):
        i, j = np.random.choice(N, 2, replace=False)
        
        # Choose target mode based on compatibility-weighted probability
        compatibilities = []
        for k in range(N):
            compat = compute_braid_compatibility(modes[i], modes[j], modes[k])
            compatibilities.append(compat)
        
        # Normalize to probability distribution
        k_probs = softmax(compatibilities)
        k = np.random.choice(N, p=k_probs)
        
        # Get the specific compatibility for this fusion
        Φ_αβγ = compute_braid_compatibility(modes[i], modes[j], modes[k])
        total_compatibility += Φ_αβγ
        compatibility_count += 1
        
        # Modified detailed balance condition with braid factor
        numerator = p_star[k] * base_fission_rate
        denominator = p_star[i] * p_star[j] * base_fusion_rate * Φ_αβγ + 1e-12
        
        accept_prob = min(1, numerator / denominator)
        
        if np.random.random() < accept_prob and p[i] > 0 and p[j] > 0:
            # Perform fusion
            total_mass = p[i] + p[j]
            p[k] += total_mass
            p[i] = 0
            p[j] = 0
            t_fusion_events += 1

    # Fission process: B_γ → B_α ⊗ B_β with braid constraints
    for _ in range(int(base_fission_rate * N)):
        k = np.random.choice(N)
        if p[k] > 0.01:  # Only split modes with significant probability
            # Choose split modes based on reverse compatibility
            reverse_compatibilities = []
            for i in range(N):
                for j in range(i+1, N):
                    rev_compat = compute_reverse_braid_compatibility(modes[k], modes[i], modes[j])
                    reverse_compatibilities.append((i, j, rev_compat))
            
            if reverse_compatibilities:
                # Weight by reverse compatibility
                indices, compats = zip(*[(idx, comp) for idx, comp in enumerate(reverse_compatibilities)])
                compats = np.array([c for _, _, c in reverse_compatibilities])
                probs = softmax(compats)
                chosen_idx = np.random.choice(len(reverse_compatibilities), p=probs)
                i, j, Φ_γαβ_inv = reverse_compatibilities[chosen_idx]
                
                # Modified detailed balance for fission
                numerator = p_star[i] * p_star[j] * base_fusion_rate
                denominator = p_star[k] * base_fission_rate * Φ_γαβ_inv + 1e-12
                
                accept_prob = min(1, numerator / denominator)
                
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
    avg_compatibility = total_compatibility / compatibility_count if compatibility_count > 0 else 0
    
    entropy_trace.append(entropy)
    relative_entropy_trace.append(relative_entropy)
    curvature_alignment_trace.append(curvature_alignment)
    fusion_events.append(t_fusion_events)
    fission_events.append(t_fission_events)
    compatibility_trace.append(avg_compatibility)

# Enhanced Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Entropy plot
ax1.plot(entropy_trace, 'b-', linewidth=2)
ax1.set_title('Symbolic Entropy $S_U(t)$', fontsize=14)
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy')
ax1.grid(True, alpha=0.3)

# Relative entropy plot
ax2.plot(relative_entropy_trace, 'r-', linewidth=2)
ax2.set_title('Relative Entropy $D(p||p^*)$', fontsize=14)
ax2.set_xlabel('Time')
ax2.set_ylabel('$D(p||p^*)$')
ax2.grid(True, alpha=0.3)

# Curvature alignment and compatibility
ax3.plot(curvature_alignment_trace, 'g-', label='Curvature Alignment', linewidth=2)
ax3_twin = ax3.twinx()
ax3_twin.plot(compatibility_trace, 'm--', label='Braid Compatibility', alpha=0.7)
ax3.set_title('Curvature & Compatibility', fontsize=14)
ax3.set_xlabel('Time')
ax3.set_ylabel('Curvature Alignment')
ax3_twin.set_ylabel('Braid Compatibility')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Event rates
ax4.plot(fusion_events, 'c-', label='Fusion events', linewidth=2)
ax4.plot(fission_events, 'y-', label='Fission events', linewidth=2)
ax4.set_title('Fusion/Fission Events', fontsize=14)
ax4.set_xlabel('Time')
ax4.set_ylabel('Events per timestep')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final statistics
print("="*50)
print("SFIT-XSM SIMULATION RESULTS")
print("="*50)
print(f"Initial entropy: {entropy_trace[0]:.4f}")
print(f"Final entropy: {entropy_trace[-1]:.4f}")
print(f"Entropy change: {entropy_trace[-1] - entropy_trace[0]:.4f}")
print(f"Initial relative entropy: {relative_entropy_trace[0]:.4f}")
print(f"Final relative entropy: {relative_entropy_trace[-1]:.4f}")
print(f"Initial curvature alignment: {curvature_alignment_trace[0]:.4f}")
print(f"Final curvature alignment: {curvature_alignment_trace[-1]:.4f}")
print(f"Average braid compatibility: {np.mean(compatibility_trace):.4f}")
print("="*50)