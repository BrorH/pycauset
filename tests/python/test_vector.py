import unittest
import os
import shutil
import pycauset
from pycauset import Vector, save

class TestVector(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_vector_output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_float_vector(self):
        v = Vector(5, dtype="float")
        self.assertEqual(len(v), 5)
        self.assertEqual(v.shape, (5,))
        v[0] = 3.14
        self.assertAlmostEqual(v[0], 3.14)
        self.assertEqual(v[1], 0.0)

    def test_int_vector(self):
        v = Vector([1, 2, 3, 4, 5], dtype="int")
        self.assertEqual(len(v), 5)
        self.assertEqual(v[2], 3)
        v[0] = 10
        self.assertEqual(v[0], 10)

    def test_bool_vector(self):
        v = Vector([True, False, True], dtype="bool")
        self.assertEqual(len(v), 3)
        self.assertTrue(v[0])
        self.assertFalse(v[1])
        v[1] = True
        self.assertTrue(v[1])

    def test_auto_dtype(self):
        v1 = Vector([1.1, 2.2])
        # Check if it's a FloatVector (by checking repr or behavior)
        self.assertTrue("FloatVector" in str(v1))
        
        v2 = Vector([1, 2, 3])
        self.assertTrue("IntegerVector" in str(v2))
        
        v3 = Vector([True, False])
        self.assertTrue("BitVector" in str(v3))

    def test_arithmetic(self):
        v1 = Vector([1, 2, 3], dtype="float")
        v2 = Vector([4, 5, 6], dtype="float")
        
        v3 = v1 + v2
        self.assertEqual(v3[0], 5.0)
        self.assertEqual(v3[2], 9.0)
        
        v4 = v1 * 2.0
        self.assertEqual(v4[0], 2.0)
        self.assertEqual(v4[2], 6.0)
        
        v5 = 2.0 * v1
        self.assertEqual(v5[0], 2.0)
        
        # Subtraction
        v6 = v2 - v1
        self.assertEqual(v6[0], 3.0)
        self.assertEqual(v6[2], 3.0)
        
        # Scalar Addition
        v7 = v1 + 5.0
        self.assertEqual(v7[0], 6.0)
        self.assertEqual(v7[2], 8.0)
        
        v8 = 5.0 + v1
        self.assertEqual(v8[0], 6.0)
        self.assertEqual(v8[2], 8.0)
        
        # Integer arithmetic
        v_int = Vector([1, 2, 3], dtype="int")
        v_int_add = v_int + 5
        self.assertEqual(v_int_add[0], 6)
        self.assertTrue("IntegerVector" in str(v_int_add))
        
        v_int_mul = v_int * 2
        self.assertEqual(v_int_mul[0], 2)
        self.assertTrue("IntegerVector" in str(v_int_mul))
        
        d = v1.dot(v2)
        self.assertEqual(d, 32.0)
        
        import pycauset
        d2 = pycauset.dot(v1, v2)
        self.assertEqual(d2, 32.0)

    def test_mixed_arithmetic(self):
        v1 = Vector([1, 2], dtype="int")
        v2 = Vector([0.5, 0.5], dtype="float")
        
        v3 = v1 + v2
        self.assertEqual(v3[0], 1.5)
        # Result should be float vector
        self.assertTrue("FloatVector" in str(v3))

    def test_persistence(self):
        v = Vector([1, 2, 3], dtype="int")
        path = os.path.join(self.test_dir, "vec.pycauset")
        save(v, path)
        
        v.close()
        
        v2 = pycauset.load(path)
        self.assertEqual(len(v2), 3)
        self.assertEqual(v2[0], 1)
        self.assertEqual(v2[1], 2)
        self.assertEqual(v2[2], 3)
        self.assertTrue("IntegerVector" in str(v2))
        v2.close()

if __name__ == '__main__':
    unittest.main()
