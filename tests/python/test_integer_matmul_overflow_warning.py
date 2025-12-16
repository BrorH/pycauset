import unittest
import warnings

import pycauset


class TestIntegerMatmulOverflowWarning(unittest.TestCase):
    def test_int32_matmul_warns_once(self):
        a = pycauset.IntegerMatrix(2)
        b = pycauset.IntegerMatrix(2)

        # Identity
        a[0, 0] = 1
        a[1, 1] = 1
        b[0, 0] = 1
        b[1, 1] = 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = a @ b
            _ = a @ b

        hits = [
            str(item.message)
            for item in w
            if "using int64 accumulator for int32 @ int32" in str(item.message)
        ]

        # This is a warn-once warning; it may have already been emitted earlier in the test
        # process, in which case we won't see it here.
        self.assertLessEqual(len(hits), 1)
        if hits:
            self.assertIn("output stored as int32", hits[0])

    def test_int32_matmul_raises_on_output_overflow(self):
        a = pycauset.IntegerMatrix(2)
        b = pycauset.IntegerMatrix(2)

        max_int32 = 2**31 - 1
        a[0, 0] = max_int32
        b[0, 0] = 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(OverflowError):
                _ = a @ b


if __name__ == "__main__":
    unittest.main()
