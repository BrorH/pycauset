import unittest
import warnings

import pycauset


class TestBitMixedFastPaths(unittest.TestCase):
    def setUp(self) -> None:

        if not hasattr(pycauset, "_debug_clear_kernel_trace") or not hasattr(pycauset, "_debug_last_kernel_trace"):
            self.skipTest("Native debug trace helpers are not available")

        pycauset._debug_clear_kernel_trace()

    def test_bit_x_int32_matmul_uses_fastpath(self):
        n = 64
        a = pycauset.DenseMatrix(n, dtype="bool")
        b = pycauset.DenseMatrix(n, dtype="int32")

        # A row0 selects rows 0 and 1 from B
        a.set(0, 0, 1)
        a.set(0, 1, 1)

        b[0, 0] = 2
        b[1, 0] = 3
        b[0, 1] = 7
        b[1, 1] = 11

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            c = a @ b

        self.assertEqual(c[0, 0], 5)
        self.assertEqual(c[0, 1], 18)
        self.assertEqual(c[1, 0], 0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matmul.bit_x_i32", trace)

    def test_bit_x_int16_matmul_uses_fastpath(self):

        n = 64
        a = pycauset.DenseMatrix(n, dtype="bool")
        b = pycauset.DenseMatrix(n, dtype="int16")

        # A row0 selects rows 0 and 1 from B
        a.set(0, 0, 1)
        a.set(0, 1, 1)

        b[0, 0] = 2
        b[1, 0] = 3
        b[0, 1] = 7
        b[1, 1] = 11

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            c = a @ b

        self.assertEqual(c[0, 0], 5)
        self.assertEqual(c[0, 1], 18)
        self.assertEqual(c[1, 0], 0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matmul.bit_x_i16", trace)

    def test_bit_x_float64_matmul_uses_fastpath(self):
        n = 64
        a = pycauset.DenseMatrix(n, dtype="bool")
        b = pycauset.DenseMatrix(n, dtype="float64")

        a.set(0, 0, 1)
        a.set(0, 1, 1)

        b[0, 0] = 2.5
        b[1, 0] = 3.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            c = a @ b

        self.assertAlmostEqual(c[0, 0], 6.0)
        self.assertAlmostEqual(c[1, 0], 0.0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matmul.bit_x_f64", trace)

    def test_bit_x_float64_matmul_rectangular_uses_fastpath(self):
        a = pycauset.DenseMatrix(3, 4, dtype="bool")
        b = pycauset.DenseMatrix(4, 2, dtype="float64")

        # A row0 selects rows 0 and 2 from B
        a.set(0, 0, 1)
        a.set(0, 2, 1)

        b[0, 0] = 2.5
        b[2, 0] = 3.5
        b[0, 1] = 7.0
        b[2, 1] = 11.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            c = a @ b

        self.assertEqual(c.shape, (3, 2))
        self.assertAlmostEqual(c[0, 0], 6.0)
        self.assertAlmostEqual(c[0, 1], 18.0)
        self.assertAlmostEqual(c[1, 0], 0.0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matmul.bit_x_f64", trace)

    def test_bit_x_int32_matvec_uses_fastpath(self):
        n = 128
        m = pycauset.DenseMatrix(n, dtype="bool")
        v = pycauset.DenseVector(n, dtype="int32")

        # Row0 selects indices 0, 1, 64
        m.set(0, 0, 1)
        m.set(0, 1, 1)
        m.set(0, 64, 1)

        v[0] = 2
        v[1] = 3
        v[64] = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            out = m @ v

        self.assertEqual(out[0], 10)
        self.assertEqual(out[1], 0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matvec.bit_x_i32", trace)

    def test_bit_x_int16_matvec_uses_fastpath(self):

        n = 128
        m = pycauset.DenseMatrix(n, dtype="bool")
        v = pycauset.DenseVector(n, dtype="int16")

        # Row0 selects indices 0, 1, 64
        m.set(0, 0, 1)
        m.set(0, 1, 1)
        m.set(0, 64, 1)

        v[0] = 2
        v[1] = 3
        v[64] = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            out = m @ v

        self.assertEqual(out[0], 10)
        self.assertEqual(out[1], 0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matvec.bit_x_i16", trace)

    def test_bit_x_float64_matvec_uses_fastpath(self):
        n = 128
        m = pycauset.DenseMatrix(n, dtype="bool")
        v = pycauset.DenseVector(n, dtype="float64")

        m.set(0, 0, 1)
        m.set(0, 1, 1)
        m.set(0, 64, 1)

        v[0] = 2.0
        v[1] = 3.0
        v[64] = 5.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pycauset.PyCausetWarning)
            out = m @ v

        self.assertAlmostEqual(out[0], 10.0)
        self.assertAlmostEqual(out[1], 0.0)

        trace = pycauset._debug_last_kernel_trace()
        self.assertIn("cpu.matvec.bit_x_f64", trace)


if __name__ == "__main__":
    unittest.main()
