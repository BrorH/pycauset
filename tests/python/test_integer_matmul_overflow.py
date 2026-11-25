import unittest
import warnings

import pycauset


class TestIntegerMatmulOverflow(unittest.TestCase):
    def test_int32_matmul_raises_on_output_overflow(self):
        a = pycauset.DenseMatrix(2, dtype="int32")
        b = pycauset.DenseMatrix(2, dtype="int32")

        max_int32 = 2**31 - 1
        a[0, 0] = max_int32
        b[0, 0] = 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(OverflowError):
                _ = a @ b


if __name__ == "__main__":
    unittest.main()
