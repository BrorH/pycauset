import unittest
import pycauset
import os

class TestMatrixTranspose(unittest.TestCase):
    def test_dense_matrix_transpose(self):
        n = 10
        m = pycauset.FloatMatrix(n)
        for i in range(n):
            for j in range(n):
                m[i, j] = i * n + j
        
        mt = m.T
        self.assertEqual(mt.shape, (n, n))
        
        for i in range(n):
            for j in range(n):
                self.assertEqual(mt[i, j], m[j, i])
                self.assertEqual(mt[i, j], j * n + i)
        
        m.close()
        mt.close()

    def test_triangular_matrix_transpose(self):
        n = 10
        m = pycauset.TriangularFloatMatrix(n)
        # Fill upper triangle
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = i * n + j
        
        mt = m.T
        # mt should be lower triangular view
        # Accessing mt[j, i] where j > i (lower) should return m[i, j] (upper)
        
        for i in range(n):
            for j in range(i + 1, n):
                # mt is lower triangular, so we access (j, i)
                val = mt[j, i]
                self.assertEqual(val, m[i, j])
                
                # Accessing upper triangle of mt should be 0
                self.assertEqual(mt[i, j], 0.0)

        m.close()
        mt.close()

    def test_dense_bit_matrix_transpose(self):
        n = 10
        m = pycauset.DenseBitMatrix(n)
        # Set some bits
        m.set(1, 2, True)
        m.set(3, 4, True)
        
        mt = m.T
        self.assertTrue(mt.get(2, 1))
        self.assertTrue(mt.get(4, 3))
        self.assertFalse(mt.get(1, 2))
        
        m.close()
        mt.close()

    def test_triangular_bit_matrix_transpose(self):
        n = 10
        m = pycauset.TriangularBitMatrix(n)
        m.set(1, 2, True)
        m.set(3, 4, True)
        
        mt = m.T
        # mt is lower triangular
        # get(2, 1) should be True
        self.assertTrue(mt.get(2, 1))
        self.assertTrue(mt.get(4, 3))
        
        # get(1, 2) should be False (upper part of lower triangular matrix is 0)
        self.assertFalse(mt.get(1, 2))
        
        m.close()
        mt.close()

if __name__ == '__main__':
    unittest.main()
