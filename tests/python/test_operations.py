import unittest
import os
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_STORAGE_TMP = tempfile.TemporaryDirectory()
import pycauset
pycauset.set_backing_dir(_STORAGE_TMP.name)

class TestMatrixOperations(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_float_matrix_scalar_multiplication(self):
        # Regression test for crash in scalar multiplication
        n = 5
        m = pycauset.TriangularFloatMatrix(n)
        m[0, 1] = 0.5
        m[2, 4] = 1.5
        
        # This used to crash
        m2 = m * 2.0
        
        self.assertIsInstance(m2, pycauset.TriangularFloatMatrix)
        self.assertEqual(m2.shape, (n, n))
        self.assertAlmostEqual(m2[0, 1], 1.0)
        self.assertAlmostEqual(m2[2, 4], 3.0)
        self.assertEqual(m2[0, 0], 0.0)

        # Test rmul
        m3 = 3.0 * m
        self.assertAlmostEqual(m3[0, 1], 1.5)

    def test_bitwise_not_bit_matrix(self):
        n = 4
        m = pycauset.TriangularBitMatrix(n)
        m[0, 1] = 1
        
        inv = ~m
        
        self.assertIsInstance(inv, pycauset.TriangularBitMatrix)
        # (0,1) was 1, should be 0
        self.assertEqual(inv[0, 1], 0)
        # (0,2) was 0, should be 1
        self.assertEqual(inv[0, 2], 1)
        # Diagonal should still be 0 (implicit)
        self.assertEqual(inv[0, 0], 0)
        
        # Check padding bits (should be 0)
        # We can't check padding bits directly from python easily without inspecting the file
        # but we can check that the matrix behaves correctly.

    def test_bitwise_not_integer_matrix(self):
        n = 3
        m = pycauset.IntegerMatrix(n)
        m[0, 1] = 5
        
        inv = ~m
        
        self.assertIsInstance(inv, pycauset.IntegerMatrix)
        # IntegerMatrix uses int32_t. ~5 is -6.
        self.assertEqual(inv[0, 1], -6)
        # ~0 is -1
        self.assertEqual(inv[0, 2], -1)

    def test_bitwise_not_float_matrix(self):
        n = 3
        m = pycauset.TriangularFloatMatrix(n)
        m[0, 1] = 0.5
        
        inv = ~m
        
        self.assertIsInstance(inv, pycauset.TriangularFloatMatrix)
        
        # Invert on float matrix does bitwise NOT on the double representation
        val = m[0, 1]
        import struct
        val_bytes = struct.pack('d', val)
        val_int = struct.unpack('Q', val_bytes)[0]
        
        inv_val = inv[0, 1]
        inv_val_bytes = struct.pack('d', inv_val)
        inv_val_int = struct.unpack('Q', inv_val_bytes)[0]
        
        self.assertEqual(inv_val_int, ~val_int & 0xFFFFFFFFFFFFFFFF)

    def test_matrix_inverse_singular(self):
        # Triangular matrices are strictly upper triangular, so they are singular.
        # Their inverse is undefined (or raises an error).
        n = 3
        
        # TriangularBitMatrix
        m_bit = pycauset.TriangularBitMatrix(n)
        with self.assertRaises(RuntimeError):
            m_bit.invert()
            
        # TriangularFloatMatrix
        m_float = pycauset.TriangularFloatMatrix(n)
        with self.assertRaises(RuntimeError):
            m_float.invert()
            
        # IntegerMatrix (also strictly upper triangular in this implementation)
        m_int = pycauset.IntegerMatrix(n)
        with self.assertRaises(RuntimeError):
            m_int.invert()

    def test_matrix_inverse_not_implemented(self):
        # FloatMatrix is dense, but inverse is not yet implemented
        n = 3
        m_dense = pycauset.FloatMatrix(n)
        with self.assertRaises(RuntimeError):
            m_dense.invert()

    def test_toplevel_functions(self):
        n = 3
        m = pycauset.TriangularBitMatrix(n)
        m[0, 1] = 1
        
        # Test bitwise_not
        inv = pycauset.bitwise_not(m)
        self.assertEqual(inv[0, 1], 0)
        self.assertEqual(inv[0, 2], 1)
        
        # Test inverse (should raise)
        with self.assertRaises(RuntimeError):
            pycauset.invert(m)

    def test_toplevel_matmul_dense_rectangular(self):
        a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
        b_np = np.arange(12, dtype=np.float64).reshape(3, 4)

        a = pycauset.matrix(a_np)
        b = pycauset.matrix(b_np)

        out = pycauset.matmul(a, b)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(np.allclose(np.array(out), a_np @ b_np))

    def test_toplevel_matmul_vector_rules(self):
        # matrix @ vector -> vector
        a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
        v_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = pycauset.matrix(a_np)
        v = pycauset.vector(v_np)

        out = pycauset.matmul(a, v)
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.allclose(np.array(out), a_np @ v_np))

        # vector @ matrix -> row-vector semantics (matches v.T @ A)
        w_np = np.array([10.0, 20.0], dtype=np.float64)
        w = pycauset.vector(w_np)
        out2 = pycauset.matmul(w, a)
        self.assertEqual(out2.shape, (1, 3))
        self.assertTrue(np.allclose(np.array(out2), (w_np @ a_np).reshape(1, 3)))

        # vector @ vector -> scalar dot
        s = pycauset.matmul(v, v)
        self.assertIsInstance(s, float)
        self.assertAlmostEqual(s, float(v_np @ v_np))

if __name__ == '__main__':
    unittest.main()
