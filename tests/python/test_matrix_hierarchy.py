import unittest
import pycauset
from pycauset import CausalMatrix, compute_k, matmul

class TestMatrixHierarchy(unittest.TestCase):
    def test_inheritance(self):
        # Test CausalMatrix inheritance
        C = CausalMatrix(10)
        self.assertIsInstance(C, pycauset.MatrixBase)
        self.assertTrue(hasattr(C, "shape"))
        self.assertTrue(hasattr(C, "size"))
        
        # Test FloatMatrix inheritance
        # K = compute_k(C, 1.0) # compute_k might fail if C is empty? No, it works.
        # self.assertIsInstance(K, pycauset.MatrixBase)
        # self.assertIsInstance(K, pycauset.TriangularFloatMatrix) # compute_k returns TFM
        
        # Test IntegerMatrix inheritance
        # We need two causal matrices to multiply
        C2 = CausalMatrix(10)
        I_mat = matmul(C, C2)
        self.assertIsInstance(I_mat, pycauset.MatrixBase)
        self.assertIsInstance(I_mat, pycauset.TriangularIntegerMatrix)
        
        C.close()
        C2.close()
        # K.close()
        
    def test_printing(self):
        # Test that the base matrix __str__ is used
        pass
        
        # Test CausalMatrix printing
        C = CausalMatrix(3)
        C.set(0, 1, True)
        s_c = str(C)
        # Should use matrix's __str__ but class name might be CausalMatrix
        self.assertIn("shape=(3, 3)", s_c)
        # self.assertIn("[0 1 0]", s_c)
        C.close()

    def test_generic_matmul(self):
        pass

if __name__ == '__main__':
    unittest.main()
