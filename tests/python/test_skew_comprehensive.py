import unittest

raise unittest.SkipTest(
    "Skew eigenvalue solver was removed along with the legacy complex/eigen subsystem."
)

import sys
import os
import numpy as np
import time

# Add the python directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

import pycauset

class TestSkewSolverComprehensive(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Reset threads to default
        pycauset.set_num_threads(1)

    def _generate_skew(self, n):
        """Helper to generate a random real skew-symmetric matrix."""
        M = np.random.rand(n, n)
        return M - M.T

    def test_small_matrix(self):
        """Test with N=10, k=2. Ensures small matrices don't crash."""
        N = 10
        k = 2
        A_np = self._generate_skew(N)
        A_pc = pycauset.matrix(A_np)
        
        evals = pycauset.eigvals_skew(A_pc, k)
        self.assertEqual(evals.size(), k)
        
        # Verify against numpy
        evals_np = np.linalg.eigvals(A_np)
        evals_np_sorted = sorted(evals_np, key=abs, reverse=True)
        
        for i in range(k):
            self.assertAlmostEqual(abs(evals.get(i)), abs(evals_np_sorted[i]), places=5)

    def test_tiny_matrix(self):
        """Test with N=4, k=2. Very small case."""
        N = 4
        k = 2
        A_np = self._generate_skew(N)
        A_pc = pycauset.matrix(A_np)
        
        evals = pycauset.eigvals_skew(A_pc, k)
        self.assertEqual(evals.size(), k)
        
        evals_np = np.linalg.eigvals(A_np)
        evals_np_sorted = sorted(evals_np, key=abs, reverse=True)
        
        # Tolerance might need to be looser for tiny matrices with block artifacts?
        for i in range(k):
            self.assertAlmostEqual(abs(evals.get(i)), abs(evals_np_sorted[i]), places=4)


    def test_odd_dimension(self):
        """Test with odd N. Skew-symmetric matrices of odd dim must have a 0 eigenvalue."""
        N = 11
        k = N # Get all
        A_np = self._generate_skew(N)
        A_pc = pycauset.matrix(A_np)
        
        evals = pycauset.eigvals_skew(A_pc, k)
        
        # Check that we have at least one eigenvalue close to 0
        min_mag = min(abs(evals.get(i)) for i in range(evals.size()))
        self.assertLess(min_mag, 1e-9, "Odd dimension skew matrix should have a zero eigenvalue")

    def test_k_clamping(self):
        """Test requesting k > N."""
        N = 10
        k = 20 # Request more than exists
        A_np = self._generate_skew(N)
        A_pc = pycauset.matrix(A_np)
        
        evals = pycauset.eigvals_skew(A_pc, k)
        self.assertLessEqual(evals.size(), N, "Should not return more eigenvalues than N")
        # Note: The implementation might return N or slightly less depending on convergence/deflation

    def test_singular_matrix(self):
        """Test a matrix constructed to be singular (rank deficient)."""
        N = 20
        # Create a 10x10 skew block and pad with zeros
        A_small = self._generate_skew(10)
        A_np = np.zeros((N, N))
        A_np[:10, :10] = A_small
        
        A_pc = pycauset.matrix(A_np)
        
        # Requesting top 15 eigenvalues. 
        # 10 should be non-zero (from the block), rest zero.
        k = 15
        evals = pycauset.eigvals_skew(A_pc, k)
        
        # Count non-zeros
        non_zeros = sum(1 for i in range(evals.size()) if abs(evals.get(i)) > 1e-5)
        self.assertLessEqual(non_zeros, 10, "Should not find more non-zero eigenvalues than rank")

    def test_parallel_consistency(self):
        """Ensure single-threaded and multi-threaded runs produce consistent results."""
        N = 100
        k = 10
        A_np = self._generate_skew(N)
        A_pc = pycauset.matrix(A_np)
        
        # Run Sequential
        pycauset.set_num_threads(1)
        evals_seq = pycauset.eigvals_skew(A_pc, k)
        
        # Run Parallel
        pycauset.set_num_threads(4)
        evals_par = pycauset.eigvals_skew(A_pc, k)
        
        # Compare
        for i in range(k):
            val_seq = evals_seq.get(i)
            val_par = evals_par.get(i)
            # Compare magnitudes and imaginary parts
            self.assertAlmostEqual(abs(val_seq), abs(val_par), places=5)
            self.assertAlmostEqual(val_seq.imag, val_par.imag, places=5)

    def test_non_skew_input(self):
        """
        Test behavior when input is NOT skew-symmetric.
        The solver assumes A^T = -A. If we pass a symmetric matrix, 
        the algorithm might produce garbage or fail, but it shouldn't crash the process.
        """
        N = 20
        # Create Symmetric matrix
        M = np.random.rand(N, N)
        A_sym = M + M.T 
        A_pc = pycauset.matrix(A_sym)
        
        try:
            evals = pycauset.eigvals_skew(A_pc, k=5)
            # If it returns, check if they are garbage or real?
            # Skew solver assumes eigenvalues are imaginary.
            # Symmetric matrix has real eigenvalues.
            # The solver might interpret them as imaginary or fail to converge.
            print("\n[Info] Non-skew input did not crash. Result sample:", evals.get(0))
        except Exception as e:
            print(f"\n[Info] Non-skew input raised exception: {e}")
            # Exception is acceptable behavior

if __name__ == '__main__':
    unittest.main()
