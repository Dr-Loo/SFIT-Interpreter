import numpy as np

class SemanticEntropy:
    """
    Histogram-style entropy over a set of symbolic bins.

    - If sigma is None or <= 0: hard-binning via nearest-bin counts (piecewise constant),
      gradient is zero almost everywhere.
    - If sigma > 0: soft/gaussian kernel around each bin; we provide an analytic
      gradient of S = -sum_j p_j log p_j w.r.t. field samples Phi.
    """
    def __init__(self, bins, sigma=0.1, eps=1e-12):
        self.bins = np.asarray(bins, dtype=float).ravel()
        # Allow sigma=None / <=0 to mean "hard binning"
        if sigma is None:
            self.sigma = None
        else:
            s = float(sigma)
            self.sigma = None if s <= 0.0 else s
        self.eps = float(eps)

    @property
    def uses_soft(self) -> bool:
        return self.sigma is not None and self.sigma > 0.0

    def prob(self, Phi):
        """Return p over bins from field samples Phi."""
        Phi = np.asarray(Phi, dtype=float).ravel()
        m = self.bins.size
        if Phi.size == 0:
            return np.zeros(m, dtype=float)

        if not self.uses_soft:
            # Hard-binning: nearest bin
            idx = np.argmin((Phi[:, None] - self.bins[None, :])**2, axis=1)
            counts = np.bincount(idx, minlength=m).astype(float)
        else:
            # Soft/gaussian kernel weights
            diff = Phi[:, None] - self.bins[None, :]
            W = np.exp(-0.5 * (diff / self.sigma)**2)
            counts = W.sum(axis=0)  # (m,)

        total = counts.sum()
        if not np.isfinite(total) or total <= 0:
            # Fallback to uniform if degenerate
            return np.full(m, 1.0 / m, dtype=float)

        p = counts / total
        # Numerical clip for safety
        p = np.clip(p, 0.0, 1.0)
        return p

    def grad(self, Phi):
        """
        dS/dPhi for S = -sum_j p_j log p_j.

        Hard-binning -> zero gradient (piecewise constant).
        Soft kernel -> analytic gradient via chain rule.
        """
        Phi = np.asarray(Phi, dtype=float).ravel()
        n = Phi.size
        if n == 0:
            return np.zeros(0, dtype=float)

        if not self.uses_soft:
            return np.zeros_like(Phi)

        # Soft case
        diff = Phi[:, None] - self.bins[None, :]            # (n, m)
        W = np.exp(-0.5 * (diff / self.sigma)**2) + self.eps
        C = W.sum(axis=0)                                   # (m,)
        Csum = C.sum() + self.eps
        p = C / Csum
        p = np.clip(p, 1e-15, 1.0)
        logp1 = 1.0 + np.log(p)

        # dw/dPhi = -(diff / sigma^2) * W
        dw = W * (-(diff) / (self.sigma**2))                # (n, m)

        g = np.empty(n, dtype=float)
        for i in range(n):
            dw_i = dw[i, :]                                 # (m,)
            s1 = dw_i.sum()
            # dp_j/dPhi_i = (dw_ij * Csum - C_j * s1) / Csum^2
            dp = (dw_i * Csum - C * s1) / (Csum**2)
            g[i] = -np.dot(logp1, dp)

        g[~np.isfinite(g)] = 0.0
        return g

    def entropy(self, Phi) -> float:
        """Shannon entropy of the current bin probabilities."""
        p = self.prob(Phi)
        nz = p > 0
        return float(-(p[nz] * np.log(p[nz])).sum())
