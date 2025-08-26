# sfit-sgf

Lightweight numerics for M-orthonormal bases, projectors, and a Davis–Kahan sanity helper.

## Quickstart

```bash
# install (editable)
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt

## Quickstart

### α / ρ / S (weights & entropy)
```python
from sfit_sgf.core import m_orthonormal_basis, normalize_weights, shannon_entropy
import numpy as np

rng = np.random.default_rng(0)
B = m_orthonormal_basis(192, 3, rng=rng)
alpha_true = np.array([0.6, 0.3, 0.1], float)
v = B @ alpha_true
alpha = B.T @ v
rho = normalize_weights(alpha**2)   # squared amplitudes → weights
S = shannon_entropy(rho)
print(rho, S)

