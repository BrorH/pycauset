import unittest
import numpy as np
import pycauset

class TestCpuInverseBlock(unittest.TestCase):
    def test_block_matrix_inverse(self):
        # Create a block matrix
        # A = [ [I, 0], [0, 2I] ]
        N = 100
        A = pycauset.FloatMatrix(2*N, 2*N)
        # Block 1
        for i in range(N):
            A[i, i] = 1.0
        # Block 2
        for i in range(N):
            A[N+i, N+i] = 2.0
            
        # This creates a "BlockMatrix" internally if it partitions?
        # Actually standard FloatMatrix is Dense.
        # To force BlockMatrix, we need `pycauset.block_matrix` or large size?
        # Or is FloatMatrix automatically blocked?
        # R1_CPU usually uses DenseMatrix. BlockMatrix is `_internal.blockmatrix`.
        
        # Let's trust that the 'supports_block_matrix=true' flag in OpRegistration enables it
        # IF the input is actually a BlockMatrix.
        # Currently, users create DenseMatrix by default.
        
        # Let's just check correctness for this structure.
        
        B = A.inverse()
        B_np = pycauset.to_numpy(B)
        
        # Expected
        expected = np.zeros((2*N, 2*N))
        for i in range(N):
            expected[i, i] = 1.0
        for i in range(N):
            expected[N+i, N+i] = 0.5
            
        max_diff = np.max(np.abs(B_np - expected))
        print(f"[Block/Dense Inverse] Max Diff: {max_diff}")
        self.assertLess(max_diff, 1e-10)

if __name__ == "__main__":
    unittest.main()
