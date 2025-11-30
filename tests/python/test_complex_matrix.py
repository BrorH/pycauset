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

if __name__ == '__main__':
    unittest.main()
