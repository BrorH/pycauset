import unittest
import pycauset
import numpy as np
import time

class TestSolversExtended(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def create_complex_matrix(self, n, numpy_matrix):
        cm = pycauset.ComplexMatrix(n)
        for i in range(n):
            for j in range(n):
                val = numpy_matrix[i, j]
                cm.set(i, j, complex(val))
        return cm

    def verify_eig_decomposition(self, matrix_np, vals_pc, vecs_pc, tol=1e-5):
        n = matrix_np.shape[0]
        
        # Check each eigenpair
        for i in range(n):
            lam = vals_pc.get(i)
            
            # Extract eigenvector from pycauset result
            v = np.zeros(n, dtype=complex)
            for r in range(n):
                v[r] = vecs_pc.get(r, i)
            
            # Normalize v for stability in comparison if needed, but A*v should equal lam*v regardless
            # A * v
            Av = matrix_np @ v
            
            # lambda * v
            lam_v = lam * v
            
            # Check residual
            residual = np.linalg.norm(Av - lam_v)
            self.assertLess(residual, tol, f"Eigenpair {i} failed verification. Residual: {residual}")

    def test_random_complex_matrix(self):
        n = 50
        print(f"\nTesting Random Complex Matrix (N={n})...")
        
        # Generate random complex matrix
        A_real = np.random.randn(n, n)
        A_imag = np.random.randn(n, n)
        A_np = A_real + 1j * A_imag
        
        A_pc = self.create_complex_matrix(n, A_np)
        
        start = time.time()
        vals, vecs = pycauset.eig(A_pc)
        end = time.time()
        print(f"PyCauset eig time: {end - start:.4f}s")
        
        self.verify_eig_decomposition(A_np, vals, vecs)

    def test_hermitian_matrix(self):
        n = 50
        print(f"\nTesting Hermitian Matrix (N={n})...")
        
        # Generate Hermitian matrix (A = A^H)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A = A + A.conj().T
        
        A_pc = self.create_complex_matrix(n, A)
        
        vals, vecs = pycauset.eig(A_pc)
        
        # Check that eigenvalues are real (imaginary part close to 0)
        for i in range(n):
            lam = vals.get(i)
            self.assertLess(abs(lam.imag), 1e-9, f"Eigenvalue {i} of Hermitian matrix should be real. Got {lam}")
            
        self.verify_eig_decomposition(A, vals, vecs)

    def test_skew_hermitian_matrix(self):
        n = 50
        print(f"\nTesting Skew-Hermitian Matrix (N={n})...")
        
        # Generate Skew-Hermitian matrix (A = -A^H)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A = A - A.conj().T
        
        A_pc = self.create_complex_matrix(n, A)
        
        vals, vecs = pycauset.eig(A_pc)
        
        # Check that eigenvalues are purely imaginary (real part close to 0)
        for i in range(n):
            lam = vals.get(i)
            self.assertLess(abs(lam.real), 1e-9, f"Eigenvalue {i} of Skew-Hermitian matrix should be purely imaginary. Got {lam}")
            
        self.verify_eig_decomposition(A, vals, vecs)

if __name__ == '__main__':
    unittest.main()
