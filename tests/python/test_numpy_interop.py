import unittest
import numpy as np
import pycauset
import os

class TestNumpyInterop(unittest.TestCase):
    def setUp(self):
        pass

    def test_asarray_vector(self):
        arr = np.array([1.0, 2.0, 3.0])
        vec = pycauset.vector(arr)
        self.assertTrue(isinstance(vec, pycauset.DenseVector) and vec.dtype == "float64")
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec[0], 1.0)

        self.assertEqual(vec[1], 2.0)
        self.assertEqual(vec[2], 3.0)

        arr_int = np.array([1, 2, 3], dtype=np.int32)
        vec_int = pycauset.vector(arr_int)
        self.assertTrue(isinstance(vec_int, pycauset.DenseVector) and vec_int.dtype == "int32")
        self.assertEqual(vec_int[0], 1)

        arr_i16 = np.array([1, 2, -3], dtype=np.int16)
        vec_i16 = pycauset.vector(arr_i16)
        self.assertTrue(isinstance(vec_i16, pycauset.DenseVector) and vec_i16.dtype == "int16")
        self.assertEqual(vec_i16[0], 1)
        self.assertEqual(vec_i16[2], -3)

        arr_bool = np.array([True, False, True], dtype=bool)
        vec_bool = pycauset.vector(arr_bool)
        self.assertTrue(isinstance(vec_bool, pycauset.DenseVector) and vec_bool.dtype == "bool")
        self.assertEqual(vec_bool[0], True)
        self.assertEqual(vec_bool[1], False)

    def test_vector_from_negative_stride_numpy(self):
        arr = np.arange(10, dtype=np.float64)[::-1]
        vec = pycauset.vector(arr)
        self.assertEqual(len(vec), len(arr))
        self.assertTrue(np.array_equal(pycauset.to_numpy(vec), arr))

    def test_vector_import_keeps_float32(self):
        # F-056: float32 vectors used to be promoted to float64 silently.
        arr = np.array([1.25, -2.5, 3.0], dtype=np.float32)
        vec = pycauset.vector(arr)
        self.assertTrue(isinstance(vec, pycauset.DenseVector) and vec.dtype == "float32")
        out = pycauset.to_numpy(vec)
        self.assertEqual(out.dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(out, arr))

    def test_vector_import_complex(self):
        for dt, name in [(np.complex64, "complex_float32"), (np.complex128, "complex_float64")]:
            arr = np.array([1 + 2j, -3j], dtype=dt)
            vec = pycauset.vector(arr)
            self.assertEqual(vec.dtype, name)
            self.assertTrue(np.array_equal(pycauset.to_numpy(vec), arr))

    def test_vector_numpy_respects_max_in_ram_bytes(self):
        arr = np.arange(8, dtype=np.float64)
        native = getattr(pycauset, "_native", None)
        if native is None or not hasattr(native, "asarray"):
            self.skipTest("native.asarray not available")

        calls = {"count": 0}
        original = native.asarray

        def wrapped(x, *args, **kwargs):
            calls["count"] += 1
            return original(x, *args, **kwargs)

        native.asarray = wrapped
        try:
            vec = pycauset.vector(arr, max_in_ram_bytes=1)
            self.assertGreater(calls["count"], 0)
            self.assertEqual(len(vec), len(arr))

            # Phase 2: exporting file-backed/out-of-core objects via NumPy coercion
            # is blocked because NumPy can't pass allow_huge.
            with self.assertRaises(RuntimeError):
                np.asarray(vec)

            # Explicit override path.
            arr2 = pycauset.to_numpy(vec, allow_huge=True)
            self.assertTrue(np.array_equal(arr2, arr))
        finally:
            native.asarray = original

    def test_asarray_matrix(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat = pycauset.matrix(arr)
        self.assertTrue(isinstance(mat, pycauset.DenseMatrix) and mat.dtype == "float64")
        self.assertEqual(mat.rows(), 2)
        self.assertEqual(mat.cols(), 2)
        self.assertEqual(mat.size(), 4)
        self.assertEqual(mat[0, 0], 1.0)
        self.assertEqual(mat[1, 1], 4.0)

        arr_rect = np.arange(6, dtype=np.float64).reshape(2, 3)
        mat_rect = pycauset.matrix(arr_rect)
        self.assertTrue(isinstance(mat_rect, pycauset.DenseMatrix) and mat_rect.dtype == "float64")
        self.assertEqual(mat_rect.rows(), 2)
        self.assertEqual(mat_rect.cols(), 3)
        self.assertTrue(np.array_equal(np.asarray(mat_rect), arr_rect))

        arr_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        mat_int = pycauset.matrix(arr_int)
        self.assertTrue(isinstance(mat_int, pycauset.DenseMatrix) and mat_int.dtype == "int32")

        arr_i16 = np.array([[1, 2], [-3, 4]], dtype=np.int16)
        mat_i16 = pycauset.matrix(arr_i16)
        self.assertTrue(isinstance(mat_i16, pycauset.DenseMatrix) and mat_i16.dtype == "int16")
        self.assertEqual(mat_i16[1, 0], -3)

        arr_bool = np.array([[True, False], [False, True]], dtype=bool)
        mat_bool = pycauset.matrix(arr_bool)
        self.assertTrue(isinstance(mat_bool, pycauset.DenseMatrix) and mat_bool.dtype == "bool")

        arr_bool_rect = np.array([[True, False, True], [False, True, False]], dtype=bool)
        mat_bool_rect = pycauset.matrix(arr_bool_rect)
        self.assertTrue(isinstance(mat_bool_rect, pycauset.DenseMatrix) and mat_bool_rect.dtype == "bool")
        self.assertEqual(mat_bool_rect.rows(), 2)
        self.assertEqual(mat_bool_rect.cols(), 3)
        self.assertTrue(np.array_equal(np.asarray(mat_bool_rect), arr_bool_rect))

    def test_matrix_from_negative_stride_numpy(self):
        base = np.arange(12, dtype=np.float64).reshape(3, 4)
        arr = base[:, ::-1]
        mat = pycauset.matrix(arr)
        self.assertEqual(mat.rows(), arr.shape[0])
        self.assertEqual(mat.cols(), arr.shape[1])
        self.assertTrue(np.array_equal(pycauset.to_numpy(mat), arr))

    def test_matrix_import_float32_preserves_float32(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        mat = pycauset.matrix(arr)
        self.assertTrue(isinstance(mat, pycauset.DenseMatrix) and mat.dtype == "float32")
        out = pycauset.to_numpy(mat)
        self.assertEqual(out.dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(out, arr))

    def test_matrix_from_transposed_numpy(self):
        base = np.arange(12, dtype=np.float64).reshape(3, 4)
        arr = base.T
        self.assertFalse(arr.flags["C_CONTIGUOUS"])
        mat = pycauset.matrix(arr)
        self.assertEqual(mat.rows(), arr.shape[0])
        self.assertEqual(mat.cols(), arr.shape[1])
        self.assertTrue(np.array_equal(pycauset.to_numpy(mat), arr))

    def test_array_protocol(self):
        vec = pycauset.DenseVector(3, dtype="float64")
        vec[0] = 1.0
        vec[1] = 2.0
        vec[2] = 3.0
        arr = np.array(vec)
        self.assertTrue(np.array_equal(arr, np.array([1.0, 2.0, 3.0])))

        mat = pycauset.DenseMatrix(2, dtype="float64")
        mat[0, 0] = 1.0
        mat[1, 1] = 4.0
        arr_mat = np.array(mat)
        self.assertTrue(np.array_equal(arr_mat, np.array([[1.0, 0.0], [0.0, 4.0]])))

        if getattr(pycauset, "Int16Vector", None) is not None:
            v16 = pycauset.DenseVector(3, dtype="int16")
            v16[0] = 1
            v16[1] = -2
            v16[2] = 3
            a16 = np.array(v16)
            self.assertEqual(a16.dtype, np.dtype(np.int16))
            self.assertTrue(np.array_equal(a16, np.array([1, -2, 3], dtype=np.int16)))

        if getattr(pycauset, "Int16Matrix", None) is not None:
            m16 = pycauset.DenseMatrix(2, dtype="int16")
            m16[0, 0] = 7
            m16[1, 0] = -1
            a16m = np.array(m16)
            self.assertEqual(a16m.dtype, np.dtype(np.int16))
            self.assertTrue(np.array_equal(a16m, np.array([[7, 0], [-1, 0]], dtype=np.int16)))

    def test_arithmetic_mixed(self):
        vec = pycauset.DenseVector(3, dtype="float64")
        vec[0] = 1.0
        vec[1] = 2.0
        vec[2] = 3.0
        arr = np.array([1.0, 1.0, 1.0])
        
        # Vector + Numpy Array
        res = vec + arr
        self.assertTrue(isinstance(res, pycauset.DenseVector) and res.dtype == "float64")
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
            self.assertTrue(isinstance(res2, pycauset.DenseVector) and res2.dtype == "float64")
            self.assertEqual(res2[0], 2.0)

    def test_matmul_mixed(self):
        mat = pycauset.DenseMatrix(2, dtype="float64")
        mat[0, 0] = 1.0
        mat[1, 1] = 1.0 # Identity
        
        vec_arr = np.array([1.0, 2.0])
        
        # Matrix @ Numpy Vector
        res = mat @ vec_arr
        self.assertTrue(isinstance(res, pycauset.DenseVector) and res.dtype == "float64")
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.0)

    def test_matmul_mixed_integers_promotes_to_int32(self):

        a = pycauset.empty((2, 2), dtype=pycauset.int16)
        b = pycauset.empty((2, 2), dtype=pycauset.int32)

        # a = identity (int16)
        a[0, 0] = 1
        a[1, 1] = 1

        # b = small ints (int32)
        b[0, 0] = 2
        b[0, 1] = 3
        b[1, 0] = 4
        b[1, 1] = 5

        c = a @ b
        self.assertTrue(isinstance(c, pycauset.DenseMatrix) and c.dtype == "int32")
        self.assertEqual(c[0, 0], 2)
        self.assertEqual(c[0, 1], 3)
        self.assertEqual(c[1, 0], 4)
        self.assertEqual(c[1, 1], 5)

        d = b @ a
        self.assertTrue(isinstance(d, pycauset.DenseMatrix) and d.dtype == "int32")
        self.assertEqual(d[0, 0], 2)
        self.assertEqual(d[0, 1], 3)
        self.assertEqual(d[1, 0], 4)
        self.assertEqual(d[1, 1], 5)

    def test_dtype_tokens_matrix_and_vector(self):

        # pycauset.* dtype tokens
        m = pycauset.empty((3, 3), dtype=pycauset.int16)
        self.assertTrue(isinstance(m, pycauset.DenseMatrix) and m.dtype == "int16")
        v = pycauset.empty(3, dtype=pycauset.int16)
        self.assertTrue(isinstance(v, pycauset.DenseVector) and v.dtype == "int16")

        # NumPy dtypes
        m2 = pycauset.empty((3, 3), dtype=np.int16)
        self.assertTrue(isinstance(m2, pycauset.DenseMatrix) and m2.dtype == "int16")
        v2 = pycauset.empty(3, dtype=np.int16)
        self.assertTrue(isinstance(v2, pycauset.DenseVector) and v2.dtype == "int16")

        # Case-insensitive strings
        m3 = pycauset.empty((3, 3), dtype="INT16")
        self.assertTrue(isinstance(m3, pycauset.DenseMatrix) and m3.dtype == "int16")
        v3 = pycauset.empty(3, dtype="iNt16")
        self.assertTrue(isinstance(v3, pycauset.DenseVector) and v3.dtype == "int16")

        # Other dtypes: float32 and bool
        if getattr(pycauset, "Float32Matrix", None) is not None:
            mf32 = pycauset.empty((3, 3), dtype=pycauset.float32)
            self.assertTrue(isinstance(mf32, pycauset.DenseMatrix) and mf32.dtype == "float32")
        mb = pycauset.empty((3, 3), dtype="BOOL")
        self.assertTrue(isinstance(mb, pycauset.DenseMatrix) and mb.dtype == "bool")

if __name__ == '__main__':
    unittest.main()
