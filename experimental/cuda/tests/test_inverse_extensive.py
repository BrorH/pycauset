import numpy as np
import pycauset
import unittest
import sys

class TestInverseExtensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pycauset.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        # Set a small limit to force Out-of-Core logic
        pycauset.cuda.enable(memory_limit=10*1024*1024, enable_async=True)

    def verify_inverse(self, N, matrix_type="random"):
        print(f"Testing {matrix_type} matrix of size {N}x{N}")
        
        if matrix_type == "random":
            np.random.seed(42)
            A_np = np.random.rand(N, N).astype(np.float64) + np.eye(N)*2
        elif matrix_type == "identity":
            A_np = np.eye(N, dtype=np.float64)
        elif matrix_type == "diagonal":
            A_np = np.diag(np.arange(1, N+1, dtype=np.float64))
        elif matrix_type == "tridiagonal":
            A_np = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                A_np[i, i] = 2.0
                if i > 0: A_np[i, i-1] = -1.0
                if i < N-1: A_np[i, i+1] = -1.0
        
        A = pycauset.Float64Matrix(N)
        for i in range(N):
            for j in range(N):
                if A_np[i, j] != 0:
                    A[i, j] = A_np[i, j]
                    
        A_inv = A.inverse()
        A_inv_np = np.array(A_inv)
        
        I_rec = A_np @ A_inv_np
        I_ref = np.eye(N)
        
        max_diff = np.max(np.abs(I_rec - I_ref))
        print(f"  Max Error: {max_diff:.6e}")
        
        # Tolerance depends on condition number, but for these well-conditioned matrices it should be small
        self.assertLess(max_diff, 1e-6, f"Inverse failed for {matrix_type} {N}x{N}")

    def test_identity_512(self):
        self.verify_inverse(512, "identity")

    def test_diagonal_512(self):
        self.verify_inverse(512, "diagonal")
        
    def test_tridiagonal_512(self):
        self.verify_inverse(512, "tridiagonal")

    def test_random_512(self):
        self.verify_inverse(512, "random")

    def test_random_1024(self):
        self.verify_inverse(1024, "random")

if __name__ == '__main__':
    unittest.main()
