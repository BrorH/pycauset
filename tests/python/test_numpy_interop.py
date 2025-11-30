import unittest
import numpy as np
import pycauset
import os

class TestNumpyInterop(unittest.TestCase):
    def setUp(self):
        pass

    def test_asarray_vector(self):
        arr = np.array([1.0, 2.0, 3.0])
        vec = pycauset.asarray(arr)
        self.assertIsInstance(vec, pycauset.FloatVector)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec[0], 1.0)

        self.assertEqual(vec[1], 2.0)
        self.assertEqual(vec[2], 3.0)

        arr_int = np.array([1, 2, 3], dtype=np.int32)
        vec_int = pycauset.asarray(arr_int)
        self.assertIsInstance(vec_int, pycauset.IntegerVector)
        self.assertEqual(vec_int[0], 1)

        arr_bool = np.array([True, False, True], dtype=bool)
        vec_bool = pycauset.asarray(arr_bool)
        self.assertIsInstance(vec_bool, pycauset.BitVector)
        self.assertEqual(vec_bool[0], True)
        self.assertEqual(vec_bool[1], False)

    def test_asarray_matrix(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat = pycauset.asarray(arr)
        self.assertIsInstance(mat, pycauset.FloatMatrix)
        self.assertEqual(mat.size(), 2)
        self.assertEqual(mat[0, 0], 1.0)
        self.assertEqual(mat[1, 1], 4.0)

        arr_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        mat_int = pycauset.asarray(arr_int)
        self.assertIsInstance(mat_int, pycauset.IntegerMatrix)

        arr_bool = np.array([[True, False], [False, True]], dtype=bool)
        mat_bool = pycauset.asarray(arr_bool)
        self.assertIsInstance(mat_bool, pycauset.DenseBitMatrix)

    def test_array_protocol(self):
        vec = pycauset.FloatVector(3)
        vec[0] = 1.0
        vec[1] = 2.0
        vec[2] = 3.0
        arr = np.array(vec)
        self.assertTrue(np.array_equal(arr, np.array([1.0, 2.0, 3.0])))

        mat = pycauset.FloatMatrix(2)
        mat[0, 0] = 1.0
        mat[1, 1] = 4.0
        arr_mat = np.array(mat)
        self.assertTrue(np.array_equal(arr_mat, np.array([[1.0, 0.0], [0.0, 4.0]])))

    def test_arithmetic_mixed(self):
        vec = pycauset.FloatVector(3)
        vec[0] = 1.0
        vec[1] = 2.0
        vec[2] = 3.0
        arr = np.array([1.0, 1.0, 1.0])
        
        # Vector + Numpy Array
        res = vec + arr
        self.assertIsInstance(res, pycauset.FloatVector)
        self.assertEqual(res[0], 2.0)
        
        # Numpy Array + Vector (Reverse op)
        res2 = arr + vec
        # Note: Numpy might handle this by converting vec to array first if __radd__ isn't prioritized or if numpy takes precedence.
        # If numpy takes precedence, res2 will be a numpy array.
        # If pycauset takes precedence (via __array_priority__?), it might be a Vector.
        # Let's check what we get.
        if isinstance(res2, np.ndarray):
            self.assertTrue(np.array_equal(res2, np.array([2.0, 3.0, 4.0])))
        else:
            self.assertIsInstance(res2, pycauset.FloatVector)
            self.assertEqual(res2[0], 2.0)

    def test_matmul_mixed(self):
        mat = pycauset.FloatMatrix(2)
        mat[0, 0] = 1.0
        mat[1, 1] = 1.0 # Identity
        
        vec_arr = np.array([1.0, 2.0])
        
        # Matrix @ Numpy Vector
        res = mat @ vec_arr
        self.assertIsInstance(res, pycauset.FloatVector)
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.0)

if __name__ == '__main__':
    unittest.main()
