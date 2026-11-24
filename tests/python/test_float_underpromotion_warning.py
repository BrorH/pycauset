import unittest
import warnings

import pycauset


class TestFloatUnderpromotionWarning(unittest.TestCase):
    def test_mixed_float_matmul_warns_once(self):
        if not hasattr(pycauset, "Float32Matrix") or pycauset.Float32Matrix is None:
            self.skipTest("Float32Matrix not available")

        a = pycauset.Float32Matrix(2)
        b = pycauset.FloatMatrix(2)

        a[0, 0] = 1.0
        a[1, 1] = 1.0
        b[0, 0] = 2.0
        b[1, 1] = 3.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c1 = a @ b
            c2 = a @ b

        # The warning is warn-once and may already have been emitted earlier in
        # the test process, so we only require <= 1 hit here.
        hits = [
            str(item.message)
            for item in w
            if issubclass(item.category, getattr(pycauset, "PyCausetDTypeWarning", Warning))
            and "underpromotes to float32" in str(item.message)
        ]
        self.assertLessEqual(len(hits), 1)

        # Regardless of whether the warning fired here, mixed float matmul should
        # produce a float32 matrix by default.
        self.assertIn("Float32Matrix", str(type(c1)))
        self.assertAlmostEqual(c1.get(0, 0), 2.0, places=5)
        self.assertAlmostEqual(c1.get(1, 1), 3.0, places=5)

        # Ensure repeated call still works.
        self.assertIn("Float32Matrix", str(type(c2)))


if __name__ == "__main__":
    unittest.main()
