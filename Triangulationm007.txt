import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Estimated tetrahedron count
n = 2
# Manual adjacency: assume asymmetric coupling due to (3,1) twist
A = np.array([[0, 1],
              [1, 0]])

# Laplacian + collapse operator
L = csgraph.laplacian(A, normed=False)
np.random.seed(7)
phi_init = np.random.rand(n)
lambda_, mu_ = 0.45, 0.25
L_collapse = L + lambda_ * np.diag(phi_init**2) - mu_ * np.eye(n)

# Collapse eigenmode
eigval, eigvec = eigsh(L_collapse, k=1, which='SM')
phi = eigvec[:, 0]
phi = (phi - phi.min()) / (phi.max() - phi.min())

# Genus fingerprint
a0 = 1.0
a1 = float(np.mean(phi))
a2 = float(np.mean(phi**2))

print(f"SFIT genus vector for m007(3,1):")
print(f"a0 = {a0:.3f}")
print(f"a1 = {a1:.3f}")
print(f"a2 = {a2:.3f}")
