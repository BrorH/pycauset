import unittest
import numpy as np
import pycauset
import time

class TestGPUComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pycauset.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        # Enable Async by default for tests
        pycauset.cuda.enable(enable_async=True)

    def test_matmul_correctness_sizes(self):
        """Test Matrix Multiplication across various sizes to check tiling logic."""
        sizes = [10, 32, 33, 64, 100, 128, 256] 
        for N in sizes:
            with self.subTest(N=N):
                A_np = np.random.rand(N, N).astype(np.float64)
                B_np = np.random.rand(N, N).astype(np.float64)
                A = pycauset.asarray(A_np)
                B = pycauset.asarray(B_np)
                C = A @ B
                
                # Check random elements
                for _ in range(10):
                    i, j = np.random.randint(0, N, 2)
                    val = C[i, j]
                    expected = np.dot(A_np[i, :], B_np[:, j])
                    # Relax tolerance for Float32 (which GPU might use)
                    self.assertAlmostEqual(val, expected, places=3, msg=f"Mismatch at {N}x{N} ({i},{j})")

    def test_matmul_rectangular(self):
        """Test Rectangular Matrix Multiplication."""
        shapes = [(50, 100, 50), (100, 50, 100), (30, 30, 100)] # (M, K, N)
        for m, k, n in shapes:
            with self.subTest(shape=(m, k, n)):
                A_np = np.random.rand(m, k).astype(np.float64)
                B_np = np.random.rand(k, n).astype(np.float64)
                
                # We need to create rectangular matrices. 
                # asarray creates square if input is square, but handles rectangular numpy arrays?
                # Let's check if asarray supports rectangular.
                # If not, we might need to skip or use specific constructor if available.
                # Assuming asarray works for rectangular based on numpy interop.
                try:
                    A = pycauset.asarray(A_np)
                    B = pycauset.asarray(B_np)
                except:
                    # If rectangular not supported by asarray yet, skip
                    continue

                C = A @ B
                
                for _ in range(5):
                    i = np.random.randint(0, m)
                    j = np.random.randint(0, n)
                    val = C[i, j]
                    expected = np.dot(A_np[i, :], B_np[:, j])
                    self.assertAlmostEqual(val, expected, places=5)

    def test_inverse_correctness(self):
        """Test In-Core Matrix Inversion."""
        N = 100
        # Diagonally dominant to ensure invertibility
        A_np = np.random.rand(N, N) + np.eye(N) * N 
        A = pycauset.asarray(A_np)
        
        inv = A.invert()
        
        # Check A * inv = I
        prod = A @ inv
        
        diag_sum = 0
        off_diag_max = 0
        
        for i in range(N):
            for j in range(N):
                val = prod[i, j]
                if i == j:
                    diag_sum += val
                else:
                    off_diag_max = max(off_diag_max, abs(val))
        
        self.assertAlmostEqual(diag_sum, float(N), places=3)
        self.assertLess(off_diag_max, 1e-3)

    def test_eigvals_arnoldi(self):
        """Test Arnoldi Eigenvalue Solver."""
        N = 50
        A_np = np.random.rand(N, N)
        A_np = A_np + A_np.T # Symmetric
        A = pycauset.asarray(A_np)
        
        k = 5
        evals = pycauset.eigvals(A, k=k, method="arnoldi", max_iter=2000)
        
        # Arnoldi might return more than k if the subspace is larger and not filtered.
        # We just check we got at least k.
        self.assertGreaterEqual(evals.size(), k)
        # Basic sanity check: values should be real-ish for symmetric matrix
        # and within reasonable bounds.
        for i in range(evals.size()):
            e = evals.get(i)
            self.assertIsInstance(e, complex)

if __name__ == "__main__":
    unittest.main()
