import pycauset
import unittest
import os

class TestComplexMatrix(unittest.TestCase):
    def test_creation_and_access(self):
        n = 2
        cm = pycauset.ComplexMatrix(n)
        
        # Set values
        val = complex(1.0, 2.0)
        cm.set(0, 0, val)
        
        # Get values
        ret = cm.get(0, 0)
        self.assertEqual(ret, val)
        self.assertEqual(ret.real, 1.0)
        self.assertEqual(ret.imag, 2.0)
        
        # Check components
        self.assertEqual(cm.real.get(0, 0), 1.0)
        self.assertEqual(cm.imag.get(0, 0), 2.0)

    def test_addition(self):
        n = 2
        c1 = pycauset.ComplexMatrix(n)
        c1.set(0, 0, complex(1, 2))
        
        c2 = pycauset.ComplexMatrix(n)
        c2.set(0, 0, complex(3, 4))
        
        # (1+2i) + (3+4i) = 4+6i
        c3 = c1 + c2
        
        self.assertEqual(c3.get(0, 0), complex(4, 6))

    def test_multiplication(self):
        n = 2
        c1 = pycauset.ComplexMatrix(n)
        c1.set(0, 0, complex(1, 2)) # 1+2i
        
        c2 = pycauset.ComplexMatrix(n)
        c2.set(0, 0, complex(3, 4)) # 3+4i
        
        # (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        # Note: This is element-wise multiplication if using * operator?
        # Wait, the binding for __mul__ calls pycauset::multiply which is element-wise?
        # Let's check the implementation in ComplexMatrix.hpp.
        # It calls a.real()->multiply(*b.real()) which is element-wise multiply for DenseMatrix.
        # Yes, DenseMatrix::multiply is element-wise.
        
        c3 = c1 * c2
        self.assertEqual(c3.get(0, 0), complex(-5, 10))

    def test_conjugate(self):
        n = 2
        c1 = pycauset.ComplexMatrix(n)
        c1.set(0, 0, complex(1, 2))
        
        c_conj = c1.conjugate()
        self.assertEqual(c_conj.get(0, 0), complex(1, -2))
        
        # Check H (Hermitian) - currently just conjugate
        c_h = c1.H
        self.assertEqual(c_h.get(0, 0), complex(1, -2))

    def test_scalar_multiplication(self):
        """Test scaling ComplexMatrix by a complex scalar."""
        n = 2
        cm = pycauset.ComplexMatrix(n)
        cm.set(0, 0, 1.0 + 1.0j)
        
        scalar = 2.0 + 3.0j
        
        cm_scaled = cm * scalar
        res = cm_scaled.get(0, 0)
        # (1+i) * (2+3i) = (2-3) + (3+2)i = -1 + 5i
        self.assertAlmostEqual(res.real, -1.0)
        self.assertAlmostEqual(res.imag, 5.0)

        # Test reverse multiplication
        cm_scaled_rev = scalar * cm
        res_rev = cm_scaled_rev.get(0, 0)
        self.assertAlmostEqual(res_rev.real, -1.0)
        self.assertAlmostEqual(res_rev.imag, 5.0)

    def test_scalar_addition(self):
        """Test adding scalar to ComplexMatrix."""
        n = 2
        cm = pycauset.ComplexMatrix(n)
        cm.set(0, 0, 1.0 + 1.0j)
        
        scalar = 2.0 + 3.0j
        
        cm_added = cm + scalar
        res = cm_added.get(0, 0)
        # (1+i) + (2+3i) = 3 + 4i
        self.assertAlmostEqual(res.real, 3.0)
        self.assertAlmostEqual(res.imag, 4.0)

        # Test reverse addition
        cm_added_rev = scalar + cm
        res_rev = cm_added_rev.get(0, 0)
        self.assertAlmostEqual(res_rev.real, 3.0)
        self.assertAlmostEqual(res_rev.imag, 4.0)

if __name__ == '__main__':
    unittest.main()
