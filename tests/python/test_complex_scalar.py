import unittest
import pycauset
import os
import numpy as np
import tempfile
from pathlib import Path

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
        with tempfile.TemporaryDirectory(prefix="pycauset_test_complex_scalar_") as tmp:
            path = Path(tmp) / "test_complex_scalar.pycauset"
            m.save(str(path))

            # Load it back
            m2 = pycauset.load(str(path))
            try:
                self.assertEqual(m2.scalar, z)
            finally:
                m2.close()

        m.close()

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
