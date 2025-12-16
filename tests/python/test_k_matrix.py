import unittest
import numpy as np
import pycauset
from pycauset import CausalMatrix, compute_k

class TestKMatrix(unittest.TestCase):
    def test_small_matrix(self):
        N = 100
        a = 2.0
        
        # Create random causal matrix
        C = CausalMatrix.random(N, 0.5)
        
        # Convert to numpy for verification
        C_np = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if C.get(i, j):
                    C_np[i, j] = 1.0
                    
        # Compute K_ref = C @ inv(aI + C)
        I = np.eye(N)
        K_ref = C_np @ np.linalg.inv(a * I + C_np)
        
        # Compute K using C++ implementation
        K_mat = compute_k(C, a)
        
        # Verify
        K_test = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K_test[i, j] = K_mat.get(i, j)
        
        C.close()
        K_mat.close()
                
        self.assertTrue(np.allclose(K_ref, K_test, atol=1e-6))
        
    def test_edge_cases(self):
        N = 10
        a = 1.0
        
        # Empty matrix
        C1 = CausalMatrix(N, populate=False)
        K_mat1 = compute_k(C1, a)
        for i in range(N):
            for j in range(N):
                self.assertEqual(K_mat1.get(i, j), 0.0)
        # C1.close() # Not needed for in-memory
        # K_mat1.close()
                
        # Full upper triangular
        C2 = CausalMatrix(N, populate=False)
        for i in range(N):
            for j in range(i + 1, N):
                C2.set(i, j, True)
                
        C_np = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                C_np[i, j] = 1.0
                
        K_ref = C_np @ np.linalg.inv(a * np.eye(N) + C_np)
        
        K_mat2 = compute_k(C2, a)
        K_test = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K_test[i, j] = K_mat2.get(i, j)
        
        C2.close()
        K_mat2.close()
                
        self.assertTrue(np.allclose(K_ref, K_test, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
