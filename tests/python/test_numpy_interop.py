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
        self.assertIsInstance(vec, pycauset.FloatVector)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec[0], 1.0)

        self.assertEqual(vec[1], 2.0)
        self.assertEqual(vec[2], 3.0)

        arr_int = np.array([1, 2, 3], dtype=np.int32)
        vec_int = pycauset.vector(arr_int)
        self.assertIsInstance(vec_int, pycauset.IntegerVector)
        self.assertEqual(vec_int[0], 1)

        arr_i16 = np.array([1, 2, -3], dtype=np.int16)
        vec_i16 = pycauset.vector(arr_i16)
        if getattr(pycauset, "Int16Vector", None) is None:
            self.skipTest("Int16Vector is not available")
        self.assertIsInstance(vec_i16, pycauset.Int16Vector)
        self.assertEqual(vec_i16[0], 1)
        self.assertEqual(vec_i16[2], -3)

        arr_bool = np.array([True, False, True], dtype=bool)
        vec_bool = pycauset.vector(arr_bool)
        self.assertIsInstance(vec_bool, pycauset.BitVector)
        self.assertEqual(vec_bool[0], True)
        self.assertEqual(vec_bool[1], False)

    def test_vector_from_negative_stride_numpy(self):
        arr = np.arange(10, dtype=np.float64)[::-1]
        vec = pycauset.vector(arr)
        self.assertEqual(len(vec), len(arr))
        self.assertTrue(np.array_equal(pycauset.to_numpy(vec), arr))

    def test_vector_import_float32_promotes_to_float64(self):
        arr = np.array([1.25, -2.5, 3.0], dtype=np.float32)
        vec = pycauset.vector(arr)
        # Current contract: float32 vectors promote to float64 vectors.
        self.assertIsInstance(vec, pycauset.FloatVector)
        out = pycauset.to_numpy(vec)
        self.assertEqual(out.dtype, np.dtype(np.float64))
        self.assertTrue(np.array_equal(out, arr.astype(np.float64)))

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
        self.assertIsInstance(mat, pycauset.FloatMatrix)
        self.assertEqual(mat.rows(), 2)
        self.assertEqual(mat.cols(), 2)
        self.assertEqual(mat.size(), 4)
        self.assertEqual(mat[0, 0], 1.0)
        self.assertEqual(mat[1, 1], 4.0)

        arr_rect = np.arange(6, dtype=np.float64).reshape(2, 3)
        mat_rect = pycauset.matrix(arr_rect)
        self.assertIsInstance(mat_rect, pycauset.FloatMatrix)
        self.assertEqual(mat_rect.rows(), 2)
        self.assertEqual(mat_rect.cols(), 3)
        self.assertTrue(np.array_equal(np.asarray(mat_rect), arr_rect))

        arr_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        mat_int = pycauset.matrix(arr_int)
        self.assertIsInstance(mat_int, pycauset.IntegerMatrix)

        arr_i16 = np.array([[1, 2], [-3, 4]], dtype=np.int16)
        mat_i16 = pycauset.matrix(arr_i16)
        if getattr(pycauset, "Int16Matrix", None) is None:
            self.skipTest("Int16Matrix is not available")
        self.assertIsInstance(mat_i16, pycauset.Int16Matrix)
        self.assertEqual(mat_i16[1, 0], -3)

        arr_bool = np.array([[True, False], [False, True]], dtype=bool)
        mat_bool = pycauset.matrix(arr_bool)
        self.assertIsInstance(mat_bool, pycauset.DenseBitMatrix)

        arr_bool_rect = np.array([[True, False, True], [False, True, False]], dtype=bool)
        mat_bool_rect = pycauset.matrix(arr_bool_rect)
        self.assertIsInstance(mat_bool_rect, pycauset.DenseBitMatrix)
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
        self.assertIsInstance(mat, pycauset.Float32Matrix)
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

        if getattr(pycauset, "Int16Vector", None) is not None:
            v16 = pycauset.Int16Vector(3)
            v16[0] = 1
            v16[1] = -2
            v16[2] = 3
            a16 = np.array(v16)
            self.assertEqual(a16.dtype, np.dtype(np.int16))
            self.assertTrue(np.array_equal(a16, np.array([1, -2, 3], dtype=np.int16)))

        if getattr(pycauset, "Int16Matrix", None) is not None:
            m16 = pycauset.Int16Matrix(2)
            m16[0, 0] = 7
            m16[1, 0] = -1
            a16m = np.array(m16)
            self.assertEqual(a16m.dtype, np.dtype(np.int16))
            self.assertTrue(np.array_equal(a16m, np.array([[7, 0], [-1, 0]], dtype=np.int16)))

    def test_vector_buffer_protocol_allows_zero_copy_when_safe(self):
        vec = pycauset.FloatVector(4)
        vec[0] = 1.0
        vec[1] = 2.0
        vec[2] = 3.0
        vec[3] = 4.0

        mv = memoryview(vec)
        arr = np.asarray(mv)
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertTrue(np.array_equal(arr, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)))

        # Buffer path should be a view into the underlying storage.
        arr[1] = 123.0
        self.assertEqual(vec[1], 123.0)

        # The high-level helper should also use zero-copy when requested.
        arr2 = pycauset.to_numpy(vec, copy=False)
        arr2[2] = -5.0
        self.assertEqual(vec[2], -5.0)

    def test_vector_buffer_protocol_rejects_scaled_or_conjugated(self):
        vec = pycauset.FloatVector(2)
        vec[0] = 3.0
        vec[1] = 4.0

        # Reject raw-storage buffer export when the vector is logically scaled.
        vec.set_scalar(2.0)
        with self.assertRaises(BufferError):
            memoryview(vec)

        # Reject raw-storage buffer export when the vector is logically conjugated.
        vec2 = pycauset.FloatVector(2)
        vec2[0] = 1.0
        vec2[1] = 2.0
        vec2.set_conjugated(True)
        with self.assertRaises(BufferError):
            memoryview(vec2)

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

    def test_matmul_mixed_integers_promotes_to_int32(self):
        if getattr(pycauset, "Int16Matrix", None) is None:
            self.skipTest("Int16 types are not available")

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
        self.assertIsInstance(c, pycauset.IntegerMatrix)
        self.assertEqual(c[0, 0], 2)
        self.assertEqual(c[0, 1], 3)
        self.assertEqual(c[1, 0], 4)
        self.assertEqual(c[1, 1], 5)

        d = b @ a
        self.assertIsInstance(d, pycauset.IntegerMatrix)
        self.assertEqual(d[0, 0], 2)
        self.assertEqual(d[0, 1], 3)
        self.assertEqual(d[1, 0], 4)
        self.assertEqual(d[1, 1], 5)

    def test_dtype_tokens_matrix_and_vector(self):
        if getattr(pycauset, "Int16Matrix", None) is None or getattr(pycauset, "Int16Vector", None) is None:
            self.skipTest("Int16 types are not available")

        # pycauset.* dtype tokens
        m = pycauset.empty((3, 3), dtype=pycauset.int16)
        self.assertIsInstance(m, pycauset.Int16Matrix)
        v = pycauset.empty(3, dtype=pycauset.int16)
        self.assertIsInstance(v, pycauset.Int16Vector)

        # NumPy dtypes
        m2 = pycauset.empty((3, 3), dtype=np.int16)
        self.assertIsInstance(m2, pycauset.Int16Matrix)
        v2 = pycauset.empty(3, dtype=np.int16)
        self.assertIsInstance(v2, pycauset.Int16Vector)

        # Case-insensitive strings
        m3 = pycauset.empty((3, 3), dtype="INT16")
        self.assertIsInstance(m3, pycauset.Int16Matrix)
        v3 = pycauset.empty(3, dtype="iNt16")
        self.assertIsInstance(v3, pycauset.Int16Vector)

        # Other dtypes: float32 and bool
        if getattr(pycauset, "Float32Matrix", None) is not None:
            mf32 = pycauset.empty((3, 3), dtype=pycauset.float32)
            self.assertIsInstance(mf32, pycauset.Float32Matrix)
        mb = pycauset.empty((3, 3), dtype="BOOL")
        self.assertIsInstance(mb, pycauset.DenseBitMatrix)

if __name__ == '__main__':
    unittest.main()
