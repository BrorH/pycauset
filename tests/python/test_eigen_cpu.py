
import unittest
import pycauset
import numpy as np
import os

class TestEigenCpu(unittest.TestCase):
    def test_eigh_f64(self):
        n = 10
        # Create symmetric matrix
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T # Make symmetric
        
        A = pycauset.FloatMatrix(n, n)
        for i in range(n):
            for j in range(n):
                A.set(i, j, float(A_np[i, j]))
                
        # Run eigh
        w, V = A.eigh()
        
        # Check types
        self.assertTrue(isinstance(w, pycauset.FloatVector))
        self.assertTrue(isinstance(V, pycauset.FloatMatrix))
        
        # Check values against numpy
        w_np, V_np = np.linalg.eigh(A_np)
        
        # Eigenvalues can be slightly different order or close
        # Sort them to compare
        w_list = [w.get(i) for i in range(n)]
        w_list.sort()
        w_np.sort()
        
        # print("PyCauset vals:", w_list)
        # print("Numpy vals:", w_np)
        
        for i in range(n):
            self.assertAlmostEqual(w_list[i], w_np[i], places=5)
            
        # Check Av = lambda v for one pair
        # Actually checking that is harder with raw API, let's trust Linalg diff for now.
        
    def test_eigvalsh_f32(self):
        n = 10
        A_np = np.random.rand(n, n).astype(np.float32)
        A_np = A_np + A_np.T
        
        A = pycauset.Float32Matrix(n, n)
        for i in range(n):
            for j in range(n):
                A.set(i, j, float(A_np[i, j]))
                
        w = A.eigvalsh()
        
        self.assertTrue(isinstance(w, pycauset.Float32Vector)) # Should be FloatVector? No, Float32Vector
        
        w_np = np.linalg.eigvalsh(A_np)
        w_list = [w.get(i) for i in range(n)]
        w_list.sort()
        w_np.sort()
        
        for i in range(n):
            self.assertTrue(abs(w_list[i] - w_np[i]) < 1e-3)

    def test_view_support(self):
        # Create 10x10 symmetric matrix
        A = pycauset.FloatMatrix(10, 10)
        np_A = np.random.randn(10, 10).astype(np.float32)
        np_A = np_A + np_A.T # Symmetric
        
        for i in range(10): 
            for j in range(10): 
                A.set(i, j, float(np_A[i, j]))
        
        # Take 5x5 view
        B = A[0:5, 0:5]
        
        # Should NOT raise
        w, v = B.eigh()
        
        self.assertEqual(w.size(), 5)
        self.assertEqual(v.rows(), 5)
        
        # Check values against numpy on the slice
        B_np = np_A[0:5, 0:5]
        w_np = np.linalg.eigvalsh(B_np)
        
        w_list = [w.get(i) for i in range(5)]
        w_list.sort()
        w_np.sort()
        
        for i in range(5):
            self.assertTrue(abs(w_list[i] - w_np[i]) < 1e-4)

if __name__ == '__main__':
    unittest.main()
