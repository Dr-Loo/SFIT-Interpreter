import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Generate synthetic data reflecting SFIT/ALP differences
def generate_data(n_events=10000):
    """Creates feature space for:
    - SFIT: MET/√HT, WW mass, GW SNR, forward gap
    - ALPs: E_γγ, vertex displacement, beam halo"""
    
    # SFIT-like events (class 1)
    sfits = np.column_stack([
        np.random.normal(1.2, 0.3, n_events//2),  # MET/√HT
        np.random.normal(75, 2, n_events//2),     # m_ww [GeV]
        np.random.exponential(5, n_events//2),    # GW SNR
        np.random.uniform(0, 5, n_events//2)      # Forward gap [GeV]
    ])
    
    # ALP-like events (class 0)
    alps = np.column_stack([
        np.random.normal(0.3, 0.1, n_events//2),
        np.random.normal(0, 5, n_events//2),      # No mass peak
        np.random.exponential(0.5, n_events//2),  # Lower GW correlation
        np.random.uniform(8, 20, n_events//2)     # Larger forward energy
    ])
    
    X = np.vstack([sfits, alps])
    y = np.array([1]*(n_events//2) + [0]*(n_events//2))
    return X, y

X, y = generate_data()