import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Known tetrahedron count for m003
n = 2
A = np.array([[0, 1],
              [1, 0]])  # Manual adjacency (each connected to the other)

# Build Laplacian and collapse operator
L = csgraph.laplacian(A, normed=False)
np.random.seed(42)
phi_init = np.random.rand(n)
lambda_, mu_ = 0.4, 0.2
L_collapse = L + lambda_ * np.diag(phi_init**2) - mu_ * np.eye(n)

# Solve for collapse mode
eigval, eigvec = eigsh(L_collapse, k=1, which='SM')
phi = eigvec[:, 0]
phi = (phi - phi.min()) / (phi.max() - phi.min())

# Compute SFIT genus coefficients
a0 = 1.0
a1 = float(np.mean(phi))
a2 = float(np.mean(phi**2))

print(f"SFIT genus vector for m003 (manual adjacency):")
print(f"a0 = {a0:.3f}")
print(f"a1 = {a1:.3f}")
print(f"a2 = {a2:.3f}")
