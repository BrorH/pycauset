import unittest
import pycauset
import os

class TestMatmulTranspose(unittest.TestCase):
    def test_dense_matmul_transpose(self):
        n = 2
        A = pycauset.FloatMatrix(n)
        # [[1, 2], [3, 4]]
        A[0, 0] = 1.0
        A[0, 1] = 2.0
        A[1, 0] = 3.0
        A[1, 1] = 4.0
        
        AT = A.T
        
        # Expected A * AT
        # [[1, 2], [3, 4]] * [[1, 3], [2, 4]]
        # Row 0: 1*1 + 2*2 = 5
        # Row 0,1: 1*3 + 2*4 = 11
        # Row 1,0: 3*1 + 4*2 = 11
        # Row 1,1: 3*3 + 4*4 = 9 + 16 = 25
        
        C = A.multiply(AT)
        
        print(f"C[0,0] = {C[0,0]}")
        print(f"C[0,1] = {C[0,1]}")
        print(f"C[1,0] = {C[1,0]}")
        print(f"C[1,1] = {C[1,1]}")
        
        self.assertAlmostEqual(C[0, 0], 5.0)
        self.assertAlmostEqual(C[0, 1], 11.0)
        self.assertAlmostEqual(C[1, 0], 11.0)
        self.assertAlmostEqual(C[1, 1], 25.0)
        
        A.close()
        AT.close()
        C.close()

if __name__ == '__main__':
    unittest.main()
