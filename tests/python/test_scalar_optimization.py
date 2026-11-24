import pycauset
import unittest
import os

class TestMatrixTypes(unittest.TestCase):
    def test_integer_matrix_addition(self):
        # Create two integer matrices
        m1 = pycauset.IntegerMatrix(2)
        m1.set(0, 0, 1)
        m1.set(1, 1, 2)
        
        m2 = pycauset.IntegerMatrix(2)
        m2.set(0, 0, 3)
        m2.set(1, 1, 4)
        
        # Add them
        m3 = m1 + m2
        
        # Check type
        self.assertIsInstance(m3, pycauset.IntegerMatrix)
        self.assertEqual(m3.get(0, 0), 4)
        self.assertEqual(m3.get(1, 1), 6)

    def test_integer_matrix_scalar_multiplication(self):
        # F-001: integer matrices never carry a lazy scalar. NumPy result dtypes:
        # int32 * int -> int32 (values scaled), int32 * float -> float64.
        m1 = pycauset.IntegerMatrix(2)
        m1.set(0, 0, 2)

        m2 = m1 * 3
        self.assertIsInstance(m2, pycauset.IntegerMatrix)
        self.assertEqual(m2.get(0, 0), 6)
        self.assertEqual(m2.get_element_as_double(0, 0), 6.0)

        m3 = m1 * 3.5
        self.assertIsInstance(m3, pycauset.FloatMatrix)
        self.assertEqual(m3.get(0, 0), 7.0)

    def test_bit_matrix_addition(self):
        m1 = pycauset.causal_matrix(2) # TriangularBitMatrix
        m1.set(0, 1, 1)
        
        m2 = pycauset.causal_matrix(2)
        m2.set(0, 1, 1)
        
        # Add them (1 + 1 = 2)
        m3 = m1 + m2
        
        # Check type - should be IntegerMatrix (Triangular)
        print(f"Type of m3: {type(m3)}")
        
        self.assertEqual(m3.get_element_as_double(0, 1), 2.0)

    def test_bit_matrix_scalar_multiplication(self):
        m1 = pycauset.causal_matrix(2)
        m1.set(0, 1, 1)
        
        # Multiply by int
        m2 = m1 * 5
        
        self.assertEqual(m2.get_element_as_double(0, 1), 5.0)
        
        # Multiply by float
        m3 = m1 * 2.5
        self.assertEqual(m3.get_element_as_double(0, 1), 2.5)

if __name__ == '__main__':
    unittest.main()
