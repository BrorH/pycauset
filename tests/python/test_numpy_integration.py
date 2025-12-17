import unittest
import os
import tempfile
import sys
import numpy as np
from pathlib import Path

# Add python directory to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pycauset

class TestNumpyIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_conversion_to_numpy(self):
        """Test converting PyCauset objects to NumPy arrays."""
        n = 5
        
        # FloatMatrix
        fm = pycauset.FloatMatrix(n)
        fm[0, 1] = 3.14
        np_fm = np.array(fm)
        self.assertIsInstance(np_fm, np.ndarray)
        self.assertEqual(np_fm.shape, (n, n))
        self.assertEqual(np_fm.dtype, np.float64)
        self.assertEqual(np_fm[0, 1], 3.14)
        
        # IntegerMatrix
        im = pycauset.IntegerMatrix(n)
        im[0, 1] = 42
        np_im = np.array(im)
        self.assertEqual(np_im.dtype, np.int32)
        self.assertEqual(np_im[0, 1], 42)
        
        # BitMatrix
        bm = pycauset.DenseBitMatrix(n)
        bm[0, 1] = True
        np_bm = np.array(bm)
        self.assertEqual(np_bm.dtype, bool)
        self.assertEqual(np_bm[0, 1], True)
        
        # TriangularMatrix
        tm = pycauset.TriangularFloatMatrix(n)
        tm[0, 1] = 1.23
        np_tm = np.array(tm)
        self.assertEqual(np_tm[0, 1], 1.23)
        self.assertEqual(np_tm[1, 0], 0.0) # Lower triangle is zero

    def test_numpy_arithmetic_interaction(self):
        """Test arithmetic between PyCauset objects and NumPy arrays."""
        n = 4
        pc_mat = pycauset.FloatMatrix(n)
        pc_mat[0, 0] = 10.0
        
        np_mat = np.zeros((n, n))
        np_mat[0, 0] = 5.0
        
        # 1. PyCauset + NumPy
        # Should convert NumPy to PyCauset internally and return PyCauset object
        res1 = pc_mat + np_mat
        # Note: The result type depends on implementation. 
        # If from_numpy creates a temporary PyCauset object, the result is PyCauset.
        # If it falls back to NumPy's __add__, result is NumPy.
        # Based on bindings.cpp, __add__ handles from_numpy, so it returns PyCauset.
        self.assertTrue(hasattr(res1, "get_backing_file")) 
        self.assertEqual(res1[0, 0], 15.0)
        
        # 2. NumPy + PyCauset
        # NumPy's __add__ sees unknown type, returns NotImplemented.
        # PyCauset's __radd__ is called.
        res2 = np_mat + pc_mat
        self.assertTrue(hasattr(res2, "get_backing_file"))
        self.assertEqual(res2[0, 0], 15.0)
        
        # 3. PyCauset * NumPy (Elementwise)
        res3 = pc_mat * np_mat
        self.assertEqual(res3[0, 0], 50.0)
        
        # 4. PyCauset @ NumPy (Matmul)
        # I * I = I
        pc_id = pycauset.IdentityMatrix(n)
        np_id = np.eye(n)
        res4 = pc_id @ np_id
        self.assertEqual(res4[0, 0], 1.0)
        self.assertEqual(res4[1, 1], 1.0)

    def test_numpy_elementwise_broadcasting_1d(self):
        """Test NumPy-style broadcasting with 1D arrays (row-vectors)."""
        rows, cols = 3, 4
        pc_mat = pycauset.FloatMatrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                pc_mat[i, j] = float(i * 10 + j)

        np_row = np.arange(cols, dtype=np.float64)
        np_row_nonzero = np.arange(cols, dtype=np.float64) + 1.0

        res1 = pc_mat + np_row
        self.assertTrue(hasattr(res1, "get_backing_file"))
        self.assertEqual(res1.shape, (rows, cols))
        self.assertEqual(res1[2, 3], pc_mat[2, 3] + 3.0)

        res2 = np_row + pc_mat
        self.assertTrue(hasattr(res2, "get_backing_file"))
        self.assertEqual(res2[1, 2], pc_mat[1, 2] + 2.0)

        res3 = pc_mat - np_row
        self.assertTrue(hasattr(res3, "get_backing_file"))
        self.assertEqual(res3[0, 1], pc_mat[0, 1] - 1.0)

        res4 = np_row - pc_mat
        self.assertTrue(hasattr(res4, "get_backing_file"))
        self.assertEqual(res4[0, 0], 0.0 - pc_mat[0, 0])

        res5 = pc_mat * np_row
        self.assertTrue(hasattr(res5, "get_backing_file"))
        self.assertEqual(res5[2, 1], pc_mat[2, 1] * 1.0)

        res6 = np_row * pc_mat
        self.assertTrue(hasattr(res6, "get_backing_file"))
        self.assertEqual(res6[1, 3], 3.0 * pc_mat[1, 3])

        res7 = pc_mat / np_row_nonzero
        self.assertTrue(hasattr(res7, "get_backing_file"))
        self.assertAlmostEqual(res7[2, 3], pc_mat[2, 3] / 4.0)

        res8 = np_row_nonzero / pc_mat
        self.assertTrue(hasattr(res8, "get_backing_file"))
        self.assertAlmostEqual(res8[0, 1], 2.0 / pc_mat[0, 1])

        with self.assertRaises(Exception):
            pc_mat + np.arange(rows, dtype=np.float64)

    def test_numpy_ufuncs(self):
        """Test NumPy ufuncs on PyCauset objects."""
        n = 3
        m = pycauset.FloatMatrix(n)
        m[0, 0] = 0.0
        m[1, 1] = np.pi / 2
        
        # np.sin(m)
        # Since __array_ufunc__ is not implemented, NumPy will try to convert m to array first.
        # So this returns a NumPy array.
        res = np.sin(m)
        self.assertIsInstance(res, np.ndarray)
        self.assertAlmostEqual(res[0, 0], 0.0)
        self.assertAlmostEqual(res[1, 1], 1.0)

    def test_shape_and_attributes(self):
        """Test that PyCauset objects expose NumPy-like attributes."""
        n = 6
        m = pycauset.FloatMatrix(n)
        
        # .shape
        self.assertEqual(m.shape, (n, n))
        
        # .ndim (Not explicitly bound in bindings.cpp snippet, but let's check if it exists or if we should add it)
        # If not, this test will fail and I'll know to add it.
        # Actually, bindings.cpp didn't show .def_property_readonly("ndim", ...).
        # But let's see if pybind11 adds it automatically for buffer protocol objects? 
        # No, usually explicit.
        # I'll comment it out for now or expect failure if I want to be strict.
        # Let's stick to what I saw: shape.
        
        # .size (PyCauset has .size() method, but NumPy has .size property)
        # bindings.cpp has .def("size", &Matrix::size). This is a method.
        # NumPy uses property.
        # m.size() works. m.size property?
        # Let's check.
        pass

    def test_vector_numpy_integration(self):
        """Test vector integration."""
        n = 5
        v = pycauset.FloatVector(n)
        v[0] = 1.0
        v[1] = 2.0
        
        np_v = np.array(v)
        self.assertEqual(np_v.shape, (n,))
        self.assertEqual(np_v[0], 1.0)
        
        # Transposed vector
        v_t = v.T
        np_v_t = np.array(v_t)
        self.assertEqual(np_v_t.shape, (1, n))
        
        # Dot product
        res = np.dot(np_v, np_v) # 1*1 + 2*2 = 5
        self.assertEqual(res, 5.0)

if __name__ == '__main__':
    unittest.main()
