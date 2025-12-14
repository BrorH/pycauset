import unittest
import os
import numpy as np
from pycauset import Matrix, Float32Matrix, eigvals_arnoldi

class TestFloatTypes(unittest.TestCase):
    def test_01_float32_matrix(self):
        print("\nTesting Float32Matrix...")
        n = 100
        m = Float32Matrix(n, "test_f32")
        m.set(0, 0, 3.14159)
        val = m.get(0, 0)
        self.assertAlmostEqual(val, 3.14159, places=5)
        
        # Test multiplication
        m2 = Float32Matrix(n, "test_f32_2")
        m2.set(0, 0, 2.0)
        
        res = m.multiply(m2, "test_f32_res")
        # res[0,0] = row 0 of m * col 0 of m2.
        # row 0 of m has 3.14 at index 0, 0 elsewhere.
        # col 0 of m2 has 2.0 at index 0, 0 elsewhere.
        # So res[0,0] should be 3.14 * 2.0 = 6.28318
        
        self.assertAlmostEqual(res.get(0, 0), 6.28318, places=4)
        
        m.close()
        m2.close()
        res.close()

    def test_03_smart_defaults(self):
        print("\nTesting Smart Defaults...")
        # Small matrix -> Float64 (standard FloatMatrix)
        m_small = Matrix(100)
        self.assertEqual(m_small.__class__.__name__, "FloatMatrix")
        m_small.close()
        
        # Medium matrix -> Float32Matrix
        # We mock the size check by forcing it or just trusting the logic.
        # Since we can't easily allocate 10k matrix in a quick test without disk usage,
        # we can check if force_precision works.
        
        m_f32 = Matrix(100, force_precision="float32")
        self.assertEqual(m_f32.__class__.__name__, "Float32Matrix")
        m_f32.close()

    def test_04_arnoldi_float32(self):
        print("\nTesting Arnoldi on Float32...")
        n = 50
        m = Float32Matrix(n, "test_arnoldi_f32")
        # Create a diagonal matrix with known eigenvalues 1..n
        for i in range(n):
            m.set(i, i, float(i + 1))
            
        evals = eigvals_arnoldi(m, k=5, max_iter=100, tol=1e-4)
        # Convert ComplexVector to numpy array
        evals_list = []
        for i in range(evals.size()):
            evals_list.append(evals.get(i))
        evals = np.array(evals_list)
        evals = np.sort(np.abs(evals))
        
        # Largest eigenvalues should be close to n, n-1, ...
        print(f"Top eigenvalues (Float32): {evals[-5:]}")
        self.assertTrue(np.abs(evals[-1] - n) < 0.5)
        m.close()

if __name__ == '__main__':
    unittest.main()
