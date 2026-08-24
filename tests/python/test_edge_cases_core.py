"""Edge-case coverage for the core linear-algebra surface.

These tests target adversarial inputs that the existing happy-path suites
don't exercise: degenerate shapes, NaN/Inf propagation, dtype promotion/
underpromotion, complex-safety, and error-by-design.

Contract references:
- documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md (dtype rules)
- documentation/internals/DType System.md (anti-promotion ethos)
"""

import unittest

import numpy as np

import pycauset as pc


def _np(m):
    """Convert a pycauset object to a NumPy array (dtype-reporting entrypoint)."""
    return pc.to_numpy(m)


class TestShapeEdges(unittest.TestCase):
    def test_zero_dim_shapes_construct(self):
        for shape in [(0, 0), (0, 3), (3, 0)]:
            m = pc.zeros(shape, dtype="float64")
            self.assertEqual(tuple(m.shape), shape)

    def test_unit_and_rectangular_shapes_construct(self):
        for shape in [(1, 1), (2, 3), (3, 2)]:
            m = pc.zeros(shape, dtype="float64")
            self.assertEqual(tuple(m.shape), shape)

    def test_rectangular_matmul_correctness(self):
        a = pc.matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # 2x3
        b = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # 3x2
        got = _np(a @ b)
        want = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) @ np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        self.assertEqual(got.shape, (2, 2))
        np.testing.assert_allclose(got, want)


class TestErrorsByDesign(unittest.TestCase):
    def test_zeros_requires_dtype(self):
        with self.assertRaises(TypeError):
            pc.zeros((2, 2))

    def test_ones_requires_dtype(self):
        with self.assertRaises(TypeError):
            pc.ones((2, 2))

    def test_empty_requires_dtype(self):
        with self.assertRaises(TypeError):
            pc.empty((2, 2))

    def test_shape_mismatch_matmul_raises(self):
        a = pc.zeros((2, 3), dtype="float64")
        b = pc.zeros((4, 5), dtype="float64")
        with self.assertRaises(ValueError):
            a @ b


class TestNaNInfPropagation(unittest.TestCase):
    def test_nan_sum_is_nan(self):
        m = pc.matrix(np.array([[np.nan, 1.0], [2.0, 3.0]]))
        self.assertTrue(np.isnan(pc.sum(m)))

    def test_nan_add_propagates(self):
        m = pc.matrix(np.array([[np.nan, 1.0], [2.0, 3.0]]))
        got = _np(m + m)
        self.assertTrue(np.isnan(got[0, 0]))
        self.assertFalse(np.isnan(got[1, 1]))

    def test_inf_roundtrips(self):
        m = pc.matrix(np.array([[np.inf, -np.inf], [0.0, 1.0]]))
        got = _np(m)
        self.assertTrue(np.isinf(got[0, 0]))
        self.assertTrue(np.isinf(got[0, 1]))
        self.assertEqual(got[1, 1], 1.0)


class TestDTypeInvariants(unittest.TestCase):
    def test_mixed_float_underpromotes_to_float32(self):
        a = pc.ones((2, 2), dtype="float32")
        b = pc.ones((2, 2), dtype="float64")
        got = a + b
        self.assertEqual(_np(got).dtype, np.dtype(np.float32))

    def test_mixed_float_underpromotion_warns(self):
        # The warning is emitted warn-once at the C++ layer, so it cannot be
        # re-triggered in the same process. Check it in a fresh subprocess.
        import subprocess
        import sys

        code = (
            "import warnings, pycauset as pc\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    pc.ones((2, 2), dtype='float32') + pc.ones((2, 2), dtype='float64')\n"
            "print([x.category.__name__ for x in w])\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertIn("PyCausetDTypeWarning", out.stdout)

    def test_complex_imag_not_dropped(self):
        z = pc.matrix(np.array([[1 + 2j, 0], [0, 1]]))
        got = _np(z)
        self.assertEqual(got.dtype, np.dtype(np.complex128))
        self.assertEqual(got[0, 0], 1 + 2j)


if __name__ == "__main__":
    unittest.main()
