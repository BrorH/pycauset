import unittest
import numpy as np
import pycauset

class TestCpuInverse(unittest.TestCase):
    def test_inverse_lapack_small(self):
        # 1. Fits in RAM (default threshold is huge, so this uses LAPACK)
        N = 100
        np.random.seed(42)
        A_np = np.random.rand(N, N) + np.eye(N) * 2.0
        A = pycauset.matrix(A_np)
        
        # Invert
        msg = "CPU Inverse (LAPACK)"
        try:
            A_inv = A.inverse()
        except AttributeError:
             # Fallback if A.inverse() is not sugar, check pycauset.invert(A)
             A_inv = pycauset.invert(A)
        
        A_inv_np = pycauset.to_numpy(A_inv)
        
        expected = np.linalg.inv(A_np)
        max_diff = np.max(np.abs(A_inv_np - expected))
        
        print(f"[{msg}] Max Diff: {max_diff}")
        self.assertLess(max_diff, 1e-10)

    # def test_inverse_too_large_throws(self):
    #     # Hard to test without allocating > RAM. 
    #     # MemoryGovernor 'anti-nanny' rule allows usage if it fits in physical RAM, ignoring soft threshold for direct paths.
    #     pass

if __name__ == "__main__":
    unittest.main()
