import unittest
import warnings

import pycauset


class TestFloatUnderpromotionWarning(unittest.TestCase):
    def test_mixed_float_matmul_warns(self):

        a = pycauset.DenseMatrix(2, dtype="float32")
        b = pycauset.DenseMatrix(2, dtype="float64")

        a[0, 0] = 1.0
        a[1, 1] = 1.0
        b[0, 0] = 2.0
        b[1, 1] = 3.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c1 = a @ b
            c2 = a @ b

        # Python's warnings filter owns deduplication: under "always" every call warns.
        hits = [
            str(item.message)
            for item in w
            if issubclass(item.category, getattr(pycauset, "PyCausetDTypeWarning", Warning))
            and "underpromotes to float32" in str(item.message)
        ]
        self.assertEqual(len(hits), 2)

        # Mixed float matmul produces float32.
        self.assertTrue(isinstance(c1, pycauset.DenseMatrix) and c1.dtype == "float32")
        self.assertAlmostEqual(c1.get(0, 0), 2.0, places=5)
        self.assertAlmostEqual(c1.get(1, 1), 3.0, places=5)

        # Ensure repeated call still works.
        self.assertTrue(isinstance(c2, pycauset.DenseMatrix) and c2.dtype == "float32")


if __name__ == "__main__":
    unittest.main()
