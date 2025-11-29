import unittest
import pycauset
from pycauset import matrix, CausalMatrix, compute_k, matmul

class TestMatrixHierarchy(unittest.TestCase):
    def test_inheritance(self):
        # Test CausalMatrix inheritance
        C = CausalMatrix(10, populate=False)
        self.assertIsInstance(C, matrix)
        self.assertTrue(hasattr(C, "shape"))
        self.assertTrue(hasattr(C, "size"))
        
        # Test FloatMatrix inheritance
        K = compute_k(C, 1.0)
        self.assertIsInstance(K, matrix)
        self.assertIsInstance(K, pycauset.FloatMatrix)
        
        # Test IntegerMatrix inheritance
        # We need two causal matrices to multiply
        C2 = CausalMatrix(10, populate=False)
        I_mat = matmul(C, C2)
        self.assertIsInstance(I_mat, matrix)
        self.assertIsInstance(I_mat, pycauset.IntegerMatrix)
        
        C.close()
        C2.close()
        K.close()
        # IntegerMatrix doesn't have close exposed in bindings? Let's check.
        # It seems it doesn't. That might be an issue for cleanup if it holds a file handle.
        
    def test_printing(self):
        # Test that the base matrix __str__ is used
        m = matrix(3, populate=False)
        m.set(0, 0, 1)
        s = str(m)
        self.assertIn("matrix(shape=(3, 3))", s)
        self.assertIn("[1 0 0]", s)
        
        # Test CausalMatrix printing
        C = CausalMatrix(3, populate=False)
        C.set(0, 1, True)
        s_c = str(C)
        # Should use matrix's __str__ but class name might be CausalMatrix
        self.assertIn("causalmatrix(shape=(3, 3))", s_c)
        self.assertIn("[0 1 0]", s_c)
        C.close()

    def test_generic_matmul(self):
        # Create two generic matrices
        A = matrix(2)
        A.set(0, 0, 1); A.set(0, 1, 2)
        A.set(1, 0, 3); A.set(1, 1, 4)
        
        B = matrix(2)
        B.set(0, 0, 2); B.set(0, 1, 0)
        B.set(1, 0, 1); B.set(1, 1, 2)
        
        # Expected:
        # [1 2] * [2 0] = [1*2+2*1 1*0+2*2] = [4 4]
        # [3 4]   [1 2]   [3*2+4*1 3*0+4*2]   [10 8]
        
        C = matmul(A, B)
        self.assertIsInstance(C, matrix)
        self.assertEqual(C.get(0, 0), 4)
        self.assertEqual(C.get(0, 1), 4)
        self.assertEqual(C.get(1, 0), 10)
        self.assertEqual(C.get(1, 1), 8)

if __name__ == '__main__':
    unittest.main()
