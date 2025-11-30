import pycauset
import unittest
import os

class TestInnerProductTypes(unittest.TestCase):
    def test_integer_inner_product(self):
        v1 = pycauset.IntegerVector(3)
        v1[0] = 1; v1[1] = 2; v1[2] = 3
        
        v2 = pycauset.IntegerVector(3)
        v2[0] = 4; v2[1] = 5; v2[2] = 6
        
        # v1 @ v2 -> Scalar
        s = v1 @ v2
        
        # Check type
        self.assertIsInstance(s, int)
        self.assertEqual(s, 32) # 4+10+18

    def test_bit_inner_product(self):
        v1 = pycauset.BitVector(3)
        v1[0] = 1; v1[1] = 1; v1[2] = 0
        
        v2 = pycauset.BitVector(3)
        v2[0] = 1; v2[1] = 0; v2[2] = 1
        
        # v1 @ v2 -> Scalar (1*1 + 1*0 + 0*1 = 1)
        s = v1 @ v2
        
        # Check type
        self.assertIsInstance(s, int)
        self.assertEqual(s, 1)

    def test_mixed_inner_product(self):
        v1 = pycauset.IntegerVector(3)
        v1[0] = 1; v1[1] = 2; v1[2] = 3
        
        v2 = pycauset.FloatVector(3)
        v2[0] = 1.5; v2[1] = 2.5; v2[2] = 3.5
        
        # v1 @ v2 -> Scalar
        s = v1 @ v2
        
        # Check type
        self.assertIsInstance(s, float)
        self.assertEqual(s, 1.5 + 5.0 + 10.5) # 17.0

if __name__ == '__main__':
    unittest.main()
