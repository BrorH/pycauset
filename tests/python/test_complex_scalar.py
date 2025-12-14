import unittest
import pycauset
import os
import numpy as np

class TestComplexScalar(unittest.TestCase):
    def test_complex_scalar_storage(self):
        n = 10
        # Create a matrix
        m = pycauset.FloatMatrix(n)
        
        # Set a complex scalar
        z = 1.5 + 2.5j
        m.scalar = z
        
        # Verify it reads back correctly
        self.assertEqual(m.scalar, z)
        
        # Verify it persists
        m.save("test_complex_scalar")
        
        # Load it back
        m2 = pycauset.load("test_complex_scalar")
        self.assertEqual(m2.scalar, z)
        
        # Clean up
        m.close()
        m2.close()
        if os.path.exists("test_complex_scalar.json"):
            os.remove("test_complex_scalar.json")
        if os.path.exists("test_complex_scalar.dat"):
            os.remove("test_complex_scalar.dat")

    def test_pauli_jordan(self):
        # Create a small field
        field = pycauset.ScalarField(n=20, mass=0.1)
        
        # Compute Pauli-Jordan
        Delta = field.pauli_jordan()
        
        # Check type
        self.assertIsInstance(Delta, pycauset.AntiSymmetricFloat64Matrix)
        
        # Check scalar is 1j
        self.assertEqual(Delta.scalar, 1j)
        
        # Check it is antisymmetric
        self.assertTrue(Delta.is_antisymmetric())

if __name__ == '__main__':
    unittest.main()
