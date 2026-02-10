
import unittest
import pycauset
import numpy as np
import os

class TestEigGeneralCpu(unittest.TestCase):
    def test_eig_real_eigenvalues(self):
        # 1. Symmetric matrix (should have real eigvals)
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A_np = np.random.randn(n, n)
        A_np = A_np + A_np.T # Symmetric
        
        for i in range(n):
            for j in range(n):
                A.set(i, j, float(A_np[i, j]))
                
        w, v = A.eig() # Uses General Solver dgeev
        
        # Eigenvalues should be complex but with 0 imaginary part
        # Or maybe our extract logic keeps them complex.
        
        # Check explicit type
        # In Bindings, we cast DenseVector<complex<double>> to VectorBase.
        # Python should see ComplexFloat64Vector if such a class exists, or vector of complex.
        
        w_np, v_np = np.linalg.eig(A_np)
        
        w_list = [w.get(i) for i in range(n)]
        
        # Sort complex numbers by real part
        w_list.sort(key=lambda x: x.real)
        w_np_sorted = np.sort(w_np) # Sorts complex? undefined.
        # Nump sort complex: real, then imag.
        w_np_idx = np.argsort(w_np.real)
        w_np_sorted = w_np[w_np_idx]
        
        # Compare Eigenvalues
        for i in range(n):
            # Allow some tolerance
            self.assertTrue(abs(w_list[i] - w_np_sorted[i]) < 1e-4)
            
        # Check Eigenvectors? 
        # Ordering matters. dgeev returns arbitrary order? Usually matches w.
        # We won't strictly check V here as matching them up is tedious without ordering guarantees.
        
    def test_eig_complex_eigenvalues(self):
        # Rotation matrix 90 degrees
        # [[0, -1], [1, 0]]
        # Eigenvalues: i, -i
        
        A = pycauset.FloatMatrix(2, 2)
        A.set(0, 0, 0.0)
        A.set(0, 1, -1.0)
        A.set(1, 0, 1.0)
        A.set(1, 1, 0.0)
        
        w = A.eigvals()
        
        self.assertEqual(w.size(), 2)
        v0 = w.get(0)
        v1 = w.get(1)
        
        # Order unpredictable
        vals = {complex(round(v0.real, 5), round(v0.imag, 5)), complex(round(v1.real, 5), round(v1.imag, 5))}
        expected = {complex(0.0, 1.0), complex(0.0, -1.0)}
        
        self.assertEqual(vals, expected)
        
    def test_eig_full(self):
        # A matrix with known complex eigvals
        # [[1, -1], [1, 1]] -> 1 +/- i
        A = pycauset.FloatMatrix(2, 2)
        A.set(0, 0, 1.0); A.set(0, 1, -1.0)
        A.set(1, 0, 1.0); A.set(1, 1, 1.0)
        
        w, v = A.eig()
        
        # Verify A*v = w*v
        # v is likely ComplexFloatMatrix
        
        for k in range(2):
            lam = w.get(k)
            # functionality for getting complex column?
            # DenseMatrix.get_element_as_complex(row, col)
            vec = np.zeros(2, dtype=complex)
            vec[0] = v.get_element_as_complex(0, k)
            vec[1] = v.get_element_as_complex(1, k)
            
            # Matmul A * vec
            # A is real here, but we can multiply.
            res = np.array([[1, -1], [1, 1]]) @ vec
            
            target = lam * vec
            
            diff = res - target
            self.assertTrue(np.linalg.norm(diff) < 1e-4)

if __name__ == '__main__':
    unittest.main()
