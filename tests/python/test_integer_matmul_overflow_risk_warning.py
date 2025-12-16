import unittest
import warnings

import pycauset


class TestIntegerMatmulOverflowRiskWarning(unittest.TestCase):
    def test_overflow_risk_preflight_warns(self):
        # Large enough to trigger the "large matmul" preflight path.
        n = 256
        a = pycauset.IntegerMatrix(n)
        b = pycauset.IntegerMatrix(n)

        # Pick values that do NOT overflow int32 for this sparse case,
        # but are large enough that the heuristic bound n*max|A|*max|B|
        # exceeds int32 and triggers the warning.
        v = 46340  # 46340^2 < 2^31-1
        a[0, 0] = v
        b[0, 0] = v

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = a @ b

        # Sanity: operation succeeded and did not overflow.
        self.assertEqual(c[0, 0], v * v)

        hits = [
            item
            for item in w
            if "matmul preflight" in str(item.message)
            and "may overflow int32 output" in str(item.message)
        ]

        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(issubclass(hits[0].category, pycauset.PyCausetOverflowRiskWarning))


if __name__ == "__main__":
    unittest.main()
