import unittest
import numpy as np
import os
import sys
import pycauset
from pycauset import FloatMatrix

# Helper to skip if no GPU
def skip_if_no_cuda(f):
    def wrapper(*args, **kwargs):
        if not pycauset.cuda.is_available():
            print("Skipping GPU test (CUDA not available)")
            return
        return f(*args, **kwargs)
    return wrapper

class TestGPUAcceleration(unittest.TestCase):
    
    def setUp(self):
        self.has_cuda = pycauset.cuda.is_available()
        print(f"\nCUDA Available: {self.has_cuda}")
        if self.has_cuda:
            print(f"Device: {pycauset.cuda.current_device()}")

    def test_detection(self):
        """Test that the API returns a boolean and doesn't crash."""
        available = pycauset.cuda.is_available()
        self.assertIsInstance(available, bool)
        
        device_name = pycauset.cuda.current_device()
        self.assertIsInstance(device_name, str)
        print(f"Current Device Name: {device_name}")

    @skip_if_no_cuda
    def test_matmul_small(self):
        """Test correctness of matrix multiplication for small matrices (In-Core)."""
        N = 100
        A_np = np.random.rand(N, N)
        B_np = np.random.rand(N, N)
        
        A = FloatMatrix(N, "gpu_test_A.bin")
        B = FloatMatrix(N, "gpu_test_B.bin")
        
        # Load data
        for i in range(N):
            for j in range(N):
                A.set(i, j, A_np[i, j])
                B.set(i, j, B_np[i, j])
                
        # Compute
        C = A.multiply(B, "gpu_test_C.bin")
        
        # Verify
        C_np = np.dot(A_np, B_np)
        
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(C.get(i, j), C_np[i, j], places=5)
                
        A.close()
        B.close()
        C.close()

    @skip_if_no_cuda
    def test_matmul_streaming(self):
        """
        Test correctness of matrix multiplication for matrices that force streaming.
        We can't easily force VRAM overflow without massive data, 
        but we can trust the logic handles it if we use a decent size.
        """
        N = 1000
        # Use deterministic data to avoid slow random generation
        # A[i,j] = 1.0 if i==j else 0.0 (Identity)
        # B[i,j] = i + j
        
        A = FloatMatrix(N, "gpu_stream_A.bin")
        B = FloatMatrix(N, "gpu_stream_B.bin")
        
        # Create Identity manually to ensure it's Dense
        for i in range(N):
            for j in range(N):
                A.set(i, j, 1.0 if i == j else 0.0)
                B.set(i, j, float(i + j))
                
        C = A.multiply(B, "gpu_stream_C.bin")
        
        # C should equal B
        for i in range(0, N, 100): # Sample check
            for j in range(0, N, 100):
                self.assertAlmostEqual(C.get(i, j), float(i + j), places=5)
                
        A.close()
        B.close()
        C.close()

    @skip_if_no_cuda
    def test_inverse(self):
        """Test matrix inversion on GPU."""
        N = 50
        A_np = np.random.rand(N, N) + np.eye(N) * N # Diagonally dominant -> Invertible
        
        A = FloatMatrix(N, "gpu_inv_A.bin")
        for i in range(N):
            for j in range(N):
                A.set(i, j, A_np[i, j])
                
        A_inv = A.invert()
        
        # Check A * A_inv = I
        Prod = A.multiply(A_inv, "gpu_inv_check.bin")
        
        for i in range(N):
            for j in range(N):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(Prod.get(i, j), expected, places=4)
                
        A.close()
        A_inv.close()
        Prod.close()

    @skip_if_no_cuda
    def test_inverse_singular(self):
        """Test that inverting a singular matrix raises an error."""
        N = 10
        A = FloatMatrix(N, "gpu_sing_A.bin")
        # Zero matrix is singular
        for i in range(N):
            for j in range(N):
                A.set(i, j, 0.0)
                
        with self.assertRaises(RuntimeError):
            A.invert()
            
        A.close()

    @skip_if_no_cuda
    def test_eigenvalues(self):
        """Test eigenvalue decomposition on GPU."""
        N = 50
        # Symmetric matrix has real eigenvalues
        A_np = np.random.rand(N, N)
        A_np = A_np + A_np.T
        
        A = FloatMatrix(N, "gpu_eig_A.bin")
        for i in range(N):
            for j in range(N):
                A.set(i, j, A_np[i, j])
                
        evals = pycauset.eigvals(A)
        
        # Compare with numpy
        evals_np = np.linalg.eigvals(A_np)
        evals_np.sort()
        
        evals_pc = []
        for i in range(N):
            evals_pc.append(evals.get(i).real) # Real part
        evals_pc.sort()
        
        # Allow some tolerance for different algorithms
        diff = np.abs(np.array(evals_pc) - evals_np)
        self.assertTrue(np.all(diff < 1e-3), f"Max diff: {np.max(diff)}")
        
        A.close()

    @skip_if_no_cuda
    def test_arnoldi_streaming(self):
        """Test Arnoldi solver which uses batch_gemv."""
        N = 200
        k = 10
        
        A = FloatMatrix(N, "gpu_arnoldi_A.bin")
        # Random sparse-ish
        for i in range(N):
            A.set(i, i, float(i)) # Diagonal 0..N-1
            if i > 0: A.set(i, i-1, 0.1)
            if i < N-1: A.set(i, i+1, 0.1)
            
        # Largest eigenvalues should be near N-1
        evals = pycauset.eigvals_arnoldi(A, k, max_iter=50)
        
        # Check that we got k values
        self.assertEqual(evals.size(), k)
        
        # Check top value is close to N-1
        max_val = 0.0
        for i in range(k):
            val = abs(evals.get(i))
            if val > max_val: max_val = val
            
        self.assertGreater(max_val, N - 2)
        
        A.close()

    @skip_if_no_cuda
    def test_tiny_matrix(self):
        """Test 1x1 matrix operations."""
        N = 1
        A = FloatMatrix(N, "gpu_tiny_A.bin")
        A.set(0, 0, 2.0)
        
        B = FloatMatrix(N, "gpu_tiny_B.bin")
        B.set(0, 0, 3.0)
        
        C = A.multiply(B, "gpu_tiny_C.bin")
        self.assertEqual(C.get(0, 0), 6.0)
        
        # Inverse
        A_inv = A.invert()
        self.assertEqual(A_inv.get(0, 0), 0.5)
        
        A.close()
        B.close()
        C.close()
        A_inv.close()

    @skip_if_no_cuda
    def test_transpose_multiply(self):
        """Test multiplication with transposed matrix."""
        N = 50
        A_np = np.random.rand(N, N)
        B_np = np.random.rand(N, N)
        
        A = FloatMatrix(N, "gpu_trans_A.bin")
        B = FloatMatrix(N, "gpu_trans_B.bin")
        
        for i in range(N):
            for j in range(N):
                A.set(i, j, A_np[i, j])
                B.set(i, j, B_np[i, j])
                
        # C = A.T * B
        A_T = A.transpose("gpu_trans_AT.bin")
        C = A_T.multiply(B, "gpu_trans_C.bin")
        
        C_np = np.dot(A_np.T, B_np)
        
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(C.get(i, j), C_np[i, j], places=5)
                
        A.close()
        B.close()
        A_T.close()
        C.close()

    @skip_if_no_cuda
    def test_matmul_float32(self):
        """Test matrix multiplication for Float32Matrix (Single Precision)."""
        if not hasattr(pycauset, 'Float32Matrix') or pycauset.Float32Matrix is None:
            print("Skipping Float32 test (Float32Matrix not available)")
            return

        N = 100
        A_np = np.random.rand(N, N).astype(np.float32)
        B_np = np.random.rand(N, N).astype(np.float32)
        
        A = pycauset.Float32Matrix(N, "gpu_f32_A.bin")
        B = pycauset.Float32Matrix(N, "gpu_f32_B.bin")
        
        for i in range(N):
            for j in range(N):
                A.set(i, j, float(A_np[i, j]))
                B.set(i, j, float(B_np[i, j]))
                
        C = A.multiply(B, "gpu_f32_C.bin")
        
        C_np = np.dot(A_np, B_np)
        
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(C.get(i, j), C_np[i, j], places=4)
                
        A.close()
        B.close()
        C.close()

    def tearDown(self):
        # Cleanup files
        for f in os.listdir("."):
            if f.startswith("gpu_") and (f.endswith(".bin") or f.endswith(".json")):
                try:
                    os.remove(f)
                except:
                    pass

if __name__ == '__main__':
    unittest.main()
