import numpy as np

class SemanticEntropy:
    """
    Smooth sector histogram over face features g(F) -> p (probabilities).
    Provides S_U and grad wrt Φ via chain rule.
    """
    def __init__(self, centers, sigma=0.2, eps=1e-12):
        self.centers = np.asarray(centers, dtype=float)
        self.sigma = float(sigma)
        self.eps = float(eps)

    def features(self, F):
        # default feature: magnitude
        return np.sqrt(F*F + self.eps)

    def probs(self, g):
        # soft bins along centers
        C = self.centers[None, :]
        W = np.exp(-0.5 * ((g[:, None] - C) / self.sigma)**2)  # shape: (n_faces, K)
        Z = np.sum(W, axis=0) + self.eps
        p = Z / np.sum(Z)
        return p, W, Z

    def S_and_gradF(self, F):
        """
        Returns S_U and ∂S_U/∂F (vector on faces).
        """
        g = self.features(F)  # (n_faces,)
        p, W, Z = self.probs(g)  # p: (K,)
        # entropy and dS/dp
        S = float(-(p * (np.log(p + self.eps))).sum())
        dS_dp = -(np.log(p + self.eps) + 1.0)  # (K,)

        # dp/dg_f
        # Z_k = sum_f W_fk,  p_k = Z_k / sum_j Z_j
        Ztot = np.sum(Z) + self.eps
        dZ_dg = (W * (self.centers[None, :] - g[:, None]) / (self.sigma**2))  # derivative of W wrt g with sign
        dZ_dg *= -1.0  # d/dg exp(-(...)) factor

        # dp_k/dg_f = (dZ_k/dg_f * Ztot - Z_k * sum_j dZ_j/dg_f) / Ztot^2
        sum_dZdg_over_k = np.sum(dZ_dg, axis=1)  # (n_faces,)
        dp_dg = (dZ_dg * Ztot - Z[None, :] * sum_dZdg_over_k[:, None]) / (Ztot**2)  # (n_faces, K)

        # dS/dg_f = sum_k dS/dp_k * dp_k/dg_f
        dS_dg = dp_dg @ dS_dp  # (n_faces,)

        # dg/ dF_f (smooth abs): F / sqrt(F^2 + eps)
        dgdF = F / np.sqrt(F*F + self.eps)

        dS_dF = dS_dg * dgdF  # chain rule
        return S, dS_dF
