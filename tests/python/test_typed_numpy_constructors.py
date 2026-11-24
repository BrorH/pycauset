import unittest

import numpy as np

import pycauset


class TestTypedNumpyConstructors(unittest.TestCase):
    def test_typed_integer_and_unsigned_vectors_from_numpy(self):
        cases = [
            ("int32", np.int32, [1, 2, -3]),
            ("int16", np.int16, [1, 2, -3]),
            ("int8", np.int8, [1, -2, 3]),
            ("int64", np.int64, [1, 2, -3]),
            ("uint8", np.uint8, [1, 2, 3]),
            ("uint16", np.uint16, [1, 2, 3]),
            ("uint32", np.uint32, [1, 2, 3]),
            ("uint64", np.uint64, [1, 2, 3]),
        ]

        for dtype, np_dtype, values in cases:
            arr = np.array(values, dtype=np_dtype)
            v = pycauset.vector(arr)
            self.assertEqual(v.dtype, dtype)
            try:
                self.assertEqual(len(v), len(values))
                for i, expected in enumerate(arr.tolist()):
                    self.assertEqual(int(v[i]), int(expected))
            finally:
                v.close()

    def test_typed_matrices_from_numpy(self):
        # Float64 matrix
        arr_f64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        m = pycauset.matrix(arr_f64, dtype="float64")
        try:
            self.assertEqual(m.rows(), 2)
            self.assertEqual(m.cols(), 2)
            self.assertTrue(np.array_equal(np.array(m), arr_f64))
        finally:
            m.close()

        # Int32 matrix
        arr_i32 = np.array([[1, 2], [-3, 4]], dtype=np.int32)
        m2 = pycauset.matrix(arr_i32, dtype="int32")
        try:
            self.assertEqual(m2.rows(), 2)
            self.assertEqual(m2.cols(), 2)
            self.assertTrue(np.array_equal(np.array(m2), arr_i32))
        finally:
            m2.close()

        # Complex128 matrix

        arr_c128 = np.array([[1 + 2j, 0 + 0j], [3 - 4j, 5 + 0j]], dtype=np.complex128)
        m3 = pycauset.matrix(arr_c128, dtype="complex_float64")
        try:
            out = np.array(m3)
            self.assertEqual(out.dtype, np.dtype(np.complex128))
            self.assertTrue(np.allclose(out, arr_c128))
        finally:
            m3.close()


if __name__ == "__main__":
    unittest.main()
