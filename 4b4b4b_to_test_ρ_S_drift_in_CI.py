import numpy as np
from scipy import sparse

class UnitTestMicrogrid:
    def __init__(self):
        """4x4x4 unit test microgrid with deterministic results"""
        self.E = 192  # 3 * 4*4*4 edges
        self.expected_alpha = np.array([0.6, 0.3, 0.1])
        self.expected_rho = self.expected_alpha**2 / np.sum(self.expected_alpha**2)
        self.expected_S = -np.sum(self.expected_rho * np.log(self.expected_rho + 1e-16))
        self.build_harmonic_basis()
        
    def build_harmonic_basis(self):
        """Create orthonormal harmonic basis vectors"""
        self.B_orth = np.zeros((self.E, 3))
        rng = np.random.RandomState(42)  # Fixed seed
        
        for i in range(3):
            vec = rng.normal(size=self.E)
            for j in range(i):  # Gram-Schmidt orthogonalization
                proj = np.dot(self.B_orth[:, j], vec)
                vec -= proj * self.B_orth[:, j]
            self.B_orth[:, i] = vec / np.linalg.norm(vec)
        
    def create_test_vector(self):
        """Create test vector with exact harmonic components"""
        v = np.zeros(self.E)
        for i in range(3):
            v += self.expected_alpha[i] * self.B_orth[:, i]
        return v
    
    def compute_memory_metrics(self, v):
        """Compute ρ, S memory metrics"""
        alpha = self.B_orth.T @ v
        rho = np.abs(alpha)**2 / np.sum(np.abs(alpha)**2)
        S = -np.sum(rho * np.log(rho + 1e-16))
        return alpha, rho, S

def test_microgrid():
    """Regression test for ρ, S metrics"""
    grid = UnitTestMicrogrid()
    v = grid.create_test_vector()
    alpha, rho, S = grid.compute_memory_metrics(v)
    
    # Verify results match expected values
    alpha_ok = np.allclose(alpha, grid.expected_alpha, atol=1e-10)
    rho_ok = np.allclose(rho, grid.expected_rho, atol=1e-10)
    entropy_ok = abs(S - grid.expected_S) < 1e-10
    
    return alpha_ok and rho_ok and entropy_ok

if __name__ == "__main__":
    success = test_microgrid()
    import sys
    sys.exit(0 if success else 1)