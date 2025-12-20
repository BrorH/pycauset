import unittest

import numpy as np

import pycauset


class TestTypedNumpyConstructors(unittest.TestCase):
    def test_typed_integer_and_unsigned_vectors_from_numpy(self):
        cases = [
            ("IntegerVector", np.int32, [1, 2, -3]),
            ("Int16Vector", np.int16, [1, 2, -3]),
            ("Int8Vector", np.int8, [1, -2, 3]),
            ("Int64Vector", np.int64, [1, 2, -3]),
            ("UInt8Vector", np.uint8, [1, 2, 3]),
            ("UInt16Vector", np.uint16, [1, 2, 3]),
            ("UInt32Vector", np.uint32, [1, 2, 3]),
            ("UInt64Vector", np.uint64, [1, 2, 3]),
        ]

        for class_name, np_dtype, values in cases:
            cls = getattr(pycauset, class_name, None)
            if cls is None:
                self.skipTest(f"{class_name} is not available")

            arr = np.array(values, dtype=np_dtype)
            v = cls(arr)
            try:
                self.assertEqual(len(v), len(values))
                for i, expected in enumerate(arr.tolist()):
                    self.assertEqual(int(v[i]), int(expected))
            finally:
                v.close()

    def test_typed_matrices_from_numpy(self):
        # Float64 matrix
        arr_f64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        m = pycauset.FloatMatrix(arr_f64)
        try:
            self.assertEqual(m.rows(), 2)
            self.assertEqual(m.cols(), 2)
            self.assertTrue(np.array_equal(np.array(m), arr_f64))
        finally:
            m.close()

        # Int32 matrix
        arr_i32 = np.array([[1, 2], [-3, 4]], dtype=np.int32)
        m2 = pycauset.IntegerMatrix(arr_i32)
        try:
            self.assertEqual(m2.rows(), 2)
            self.assertEqual(m2.cols(), 2)
            self.assertTrue(np.array_equal(np.array(m2), arr_i32))
        finally:
            m2.close()

        # Complex128 matrix
        if getattr(pycauset, "ComplexFloat64Matrix", None) is None:
            self.skipTest("ComplexFloat64Matrix is not available")

        arr_c128 = np.array([[1 + 2j, 0 + 0j], [3 - 4j, 5 + 0j]], dtype=np.complex128)
        m3 = pycauset.ComplexFloat64Matrix(arr_c128)
        try:
            out = np.array(m3)
            self.assertEqual(out.dtype, np.dtype(np.complex128))
            self.assertTrue(np.allclose(out, arr_c128))
        finally:
            m3.close()


if __name__ == "__main__":
    unittest.main()
