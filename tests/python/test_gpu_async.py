import unittest
import numpy as np
import pycauset
import time

class TestGPUAsync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            # Enable GPU with a very small memory limit to force streaming/tiling
            # 100 MB limit.
            # A 2000x2000 double matrix is ~32MB.
            # 3 matrices (A, B, C) = 96MB.
            # So 2000x2000 should barely fit or trigger tiling if overhead is counted.
            # Let's use 50MB to guarantee tiling for 2000x2000.
            pycauset.cuda.enable(memory_limit=50 * 1024 * 1024, enable_async=True)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA not available or failed to enable: {e}")

        if not pycauset.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        
        print(f"GPU Enabled: {pycauset.cuda.current_device()}")

    def test_matmul_streaming_correctness(self):
        """Test that streaming matmul produces correct results compared to CPU."""
        N = 1000
        print(f"\nTesting MatMul (N={N}) with streaming...")
        
        # Create random matrices
        A_np = np.random.rand(N, N).astype(np.float64)
        B_np = np.random.rand(N, N).astype(np.float64)
        
        A = pycauset.asarray(A_np)
        B = pycauset.asarray(B_np)
        
        # GPU Compute
        start = time.time()
        C = A @ B
        gpu_time = time.time() - start
        C_np = np.array(C)
        
        # CPU Verification (using numpy)
        C_ref = A_np @ B_np
        
        # Check error
        max_diff = np.max(np.abs(C_np - C_ref))
        print(f"Max Diff: {max_diff}")
        print(f"GPU Time: {gpu_time:.4f}s")
        
        # Relax tolerance for Float32 GPU execution vs Float64 CPU reference
        self.assertTrue(max_diff < 1e-3, f"MatMul result incorrect. Max diff: {max_diff}")

    def test_eigvals_streaming_correctness(self):
        """Test that streaming eigvals (Arnoldi) produces correct results."""
        N = 1000
        print(f"\nTesting Eigvals (N={N}) with streaming...")
        
        # Symmetric matrix for real eigenvalues
        A_np = np.random.rand(N, N).astype(np.float64)
        A_np = A_np + A_np.T
        
        A = pycauset.asarray(A_np)
        
        # GPU Compute
        start = time.time()
        # k=20 eigenvalues
        evals = pycauset.eigvals(A, k=20, method="arnoldi")
        gpu_time = time.time() - start
        
        # Convert to numpy array (ComplexVector might not be directly convertible)
        evals_np = np.array([evals.get(i) for i in range(len(evals))])
        evals_np = np.sort(evals_np.real) # Sort real parts
        
        # CPU Verification (numpy.linalg.eigvalsh returns all, we take largest magnitude)
        # Actually pycauset.eigvals returns largest magnitude by default for Arnoldi?
        # Let's just check against full solve
        evals_ref = np.linalg.eigvalsh(A_np)
        # Arnoldi typically finds largest magnitude.
        # Let's compare the largest one.
        
        max_gpu = np.max(np.abs(evals_np))
        max_ref = np.max(np.abs(evals_ref))
        
        diff = abs(max_gpu - max_ref)
        print(f"Max Eigenvalue GPU: {max_gpu}")
        print(f"Max Eigenvalue CPU: {max_ref}")
        print(f"Diff: {diff}")
        print(f"GPU Time: {gpu_time:.4f}s")
        
        # Relax tolerance for Float32 GPU execution
        self.assertTrue(diff < 1e-4, f"Eigenvalue result incorrect. Diff: {diff}")

if __name__ == '__main__':
    unittest.main()
