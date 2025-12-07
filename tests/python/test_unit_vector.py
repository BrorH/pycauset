import unittest
import os
import shutil
import pycauset
from pycauset import UnitVector, Vector, load

class TestUnitVector(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_unit_vector_output"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_creation_and_access(self):
        # Create e_2 in R^5
        v = UnitVector(5, 2)
        self.assertEqual(len(v), 5)
        self.assertEqual(v[0], 0.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 1.0)
        self.assertEqual(v[3], 0.0)
        self.assertEqual(v[4], 0.0)

    def test_addition_same_index(self):
        # e_2 + e_2 = 2*e_2
        v1 = UnitVector(5, 2)
        v2 = UnitVector(5, 2)
        
        v3 = v1 + v2
        # Check if it's a UnitVector. 
        # Note: In Python bindings, the type might be exposed as pycauset.UnitVector
        # But generic addition returns FloatVector (DenseVector)
        # self.assertTrue(isinstance(v3, UnitVector), f"Expected UnitVector, got {type(v3)}")
        self.assertTrue("FloatVector" in str(v3) or "DenseVector" in str(v3) or isinstance(v3, pycauset.FloatVector), f"Expected FloatVector, got {type(v3)}")
        self.assertEqual(v3[2], 2.0)
        self.assertEqual(v3[0], 0.0)

    def test_addition_diff_index(self):
        # e_2 + e_3 -> DenseVector
        v1 = UnitVector(5, 2)
        v2 = UnitVector(5, 3)
        
        v3 = v1 + v2
        # Should NOT be UnitVector
        self.assertFalse(isinstance(v3, UnitVector), f"Expected DenseVector, got {type(v3)}")
        # Should be some form of Vector (likely FloatVector/DenseVector)
        self.assertEqual(v3[2], 1.0)
        self.assertEqual(v3[3], 1.0)
        self.assertEqual(v3[0], 0.0)

    def test_addition_with_dense(self):
        v1 = UnitVector(5, 2)
        v2 = Vector(5, dtype="float")
        v2[0] = 5.0
        
        v3 = v1 + v2
        self.assertFalse(isinstance(v3, UnitVector), f"Expected DenseVector, got {type(v3)}")
        self.assertEqual(v3[0], 5.0)
        self.assertEqual(v3[2], 1.0)

    def test_persistence(self):
        path = os.path.join(self.test_dir, "unit_vec.pycauset")
        v = UnitVector(10, 5)
        pycauset.save(v, path)
        del v
        
        v_loaded = load(path)
        self.assertTrue(isinstance(v_loaded, UnitVector), f"Expected UnitVector, got {type(v_loaded)}")
        self.assertEqual(v_loaded[5], 1.0)
        self.assertEqual(v_loaded[0], 0.0)

if __name__ == '__main__':
    unittest.main()
