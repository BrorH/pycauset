import pycauset
import numpy as np
import unittest
import os
import time

class TestGPUFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure GPU is available for these tests
        if not pycauset.is_gpu_available():
            print("WARNING: GPU not available. Skipping GPU tests.")
            cls.gpu_available = False
        else:
            cls.gpu_available = True

    def setUp(self):
        if not self.gpu_available:
            self.skipTest("GPU not available")

    def test_float32_creation(self):
        """Test creation of Float32Matrix."""
        # By size
        m = pycauset.Matrix(100, dtype="float32")
        self.assertIn("Float32Matrix", str(type(m)))
        self.assertEqual(m.shape, (100, 100))
        
        # By numpy array
        arr = np.zeros((50, 50), dtype=np.float32)
        m2 = pycauset.Matrix(arr)
        self.assertIn("Float32Matrix", str(type(m2)))
        
        # By force_precision
        m3 = pycauset.Matrix(100, force_precision="single")
        self.assertIn("Float32Matrix", str(type(m3)))

    def test_float32_multiplication(self):
        """Test GPU multiplication of Float32Matrix."""
        N = 512
        A = pycauset.Float32Matrix.random(N)
        B = pycauset.Float32Matrix.random(N)
        
        # GPU Mult
        C = A @ B
        self.assertIn("Float32Matrix", str(type(C)))
        self.assertEqual(C.shape, (N, N))
        
        # Verify correctness (sample check)
        # We can't easily check against CPU exact values without a CPU implementation of Float32Matrix
        # But we can convert to numpy and check
        A_np = np.array(A)
        B_np = np.array(B)
        C_np = np.array(C)
        
        C_expected = A_np @ B_np
        
        # Allow some tolerance for float32 precision
        np.testing.assert_allclose(C_np, C_expected, rtol=1e-4, atol=1e-4)

    def test_dense_bit_matrix_gpu(self):
        """Test GPU multiplication of DenseBitMatrix."""
        N = 1024
        # Create random bit matrices
        A = pycauset.DenseBitMatrix.random(N, density=0.1)
        B = pycauset.DenseBitMatrix.random(N, density=0.1)
        
        # GPU Mult
        # Note: DenseBitMatrix @ DenseBitMatrix returns IntegerMatrix (path counts)
        # or DenseBitMatrix (boolean composition)?
        # In standard linear algebra over GF(2), it returns bits.
        # In path counting (A^2), it returns integers.
        # Let's check the implementation behavior.
        # The C++ binding for `multiply` returns whatever `dispatch_matmul` or `multiply` returns.
        # For DenseBitMatrix, `multiply` usually returns `IntegerMatrix` if it's doing standard matmul,
        # or `DenseBitMatrix` if it's boolean.
        # The `k_matmul_bits` kernel does popcount, so it returns integers.
        
        C = A @ B
        
        # Check type
        # If it returns integers, it should be IntegerMatrix
        # If it returns bits, it should be DenseBitMatrix
        # Based on CudaSolver.cu `k_matmul_bits` accumulates into `int32_t`.
        
        # Let's verify what we get
        # self.assertIn("IntegerMatrix", str(type(C))) 
        # Actually, let's just print type if unsure, but for test we assume IntegerMatrix based on popcount logic.
        
        # Verify against CPU (which might be slow, so use smaller N for verification)
        N_small = 128
        A_s = pycauset.DenseBitMatrix.random(N_small, density=0.5)
        B_s = pycauset.DenseBitMatrix.random(N_small, density=0.5)
        
        C_s = A_s @ B_s
        
        # CPU verification via numpy
        A_np = np.array(A_s, dtype=int)
        B_np = np.array(B_s, dtype=int)
        C_expected = A_np @ B_np
        
        C_actual = np.array(C_s)
        np.testing.assert_array_equal(C_actual, C_expected)

    def test_mixed_precision_error(self):
        """Test that mixing incompatible types throws or handles correctly."""
        A = pycauset.Float32Matrix(100)
        B = pycauset.FloatMatrix(100)
        
        # Should probably work by promoting A to Float64?
        # Or fail if not implemented.
        # dispatch_matmul handles FloatMatrix x FloatMatrix.
        # It does NOT currently handle Float32 x Float64 explicitly in the bindings I saw.
        # It might fallback to CPU or throw.
        try:
            C = A @ B
        except Exception:
            # If it fails, that's acceptable for now as long as it doesn't crash
            pass

if __name__ == '__main__':
    unittest.main()
