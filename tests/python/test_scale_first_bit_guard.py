import unittest
import warnings

import pycauset


class TestScaleFirstBitGuard(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure CPU path so the CpuSolver trace is populated.
        if hasattr(pycauset, "cuda"):
            pycauset.cuda.disable()

        if not hasattr(pycauset, "_debug_clear_kernel_trace") or not hasattr(pycauset, "_debug_last_kernel_trace"):
            self.skipTest("Native debug trace helpers are not available")

        pycauset._debug_clear_kernel_trace()

    def test_bitbit_matmul_uses_popcount_kernel(self):
        n = 64
        a = pycauset.DenseBitMatrix(n)
        b = pycauset.DenseBitMatrix(n)

        a.set(0, 0, 1)
        b.set(0, 0, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetDTypeWarning)
            _ = a @ b

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matmul.bitbit_popcount", trace)

    def test_bitbit_dot_uses_popcount_kernel(self):
        n = 256
        a = pycauset.BitVector(n)
        b = pycauset.BitVector(n)

        a.set(0, 1)
        a.set(63, 1)
        b.set(0, 1)
        b.set(63, 1)

        out = a.dot(b)

        self.assertEqual(out, 2.0)
        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.dot.bitbit_popcount", trace)


if __name__ == "__main__":
    unittest.main()
