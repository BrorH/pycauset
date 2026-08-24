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

    def test_complex_list_input_not_dropped(self):
        # Regression: a plain Python nested list of complex literals used to fall
        # through to the float branch and produce a broken abstract Matrix (with
        # no dtype, so to_numpy raised "data type '' not understood").
        z = pc.matrix([[1 + 2j, 0], [0, 1]])
        self.assertEqual(type(z).__name__, "ComplexFloat64Matrix")
        got = _np(z)
        self.assertEqual(got.dtype, np.dtype(np.complex128))
        self.assertEqual(got[0, 0], 1 + 2j)
        self.assertEqual(got[1, 1], 1 + 0j)

    def test_complex_list_rectangular_and_explicit_dtype(self):
        rect = pc.matrix([[1 + 2j, 3], [4, 5 + 6j], [7, 8]])
        self.assertEqual(type(rect).__name__, "ComplexFloat64Matrix")
        self.assertEqual(_np(rect).dtype, np.dtype(np.complex128))

        c32 = pc.matrix([[1 + 2j, 0], [0, 1]], dtype="complex_float32")
        self.assertEqual(_np(c32).dtype, np.dtype(np.complex64))


class TestDenseFactorizationsLapack(unittest.TestCase):
    """Regression coverage for the LAPACK-backed dense float64 solve/LU.

    These guard against the previous naive scalar Gaussian-elimination
    implementations (which were O(n^3) single-threaded and inconsistent with the
    float paths) being reintroduced, and pin the P@L@U reconstruction convention.
    """

    def test_solve_dense_float64_matches_numpy(self):
        rng = np.random.default_rng(20240824)
        a_np = rng.standard_normal((30, 30))
        a_np += np.eye(30) * 10.0  # well-conditioned
        b_np = rng.standard_normal((30, 4))
        x = pc.solve(pc.matrix(a_np), pc.matrix(b_np))
        self.assertTrue(np.allclose(a_np @ _np(x), b_np, atol=1e-10))

    def test_solve_dense_float64_singular_raises(self):
        a_np = np.zeros((4, 4))
        b_np = np.ones((4, 1))
        with self.assertRaises(Exception):
            pc.solve(pc.matrix(a_np), pc.matrix(b_np))

    def test_lu_dense_float64_reconstructs(self):
        rng = np.random.default_rng(7)
        a_np = rng.standard_normal((25, 25))
        p, l, u = pc.lu(pc.matrix(a_np))
        P, L, U = _np(p), _np(l), _np(u)
        self.assertTrue(np.allclose(P @ L @ U, a_np, atol=1e-10))
        self.assertTrue(np.allclose(np.diag(L), 1.0))
        self.assertTrue(np.allclose(np.tril(L), L))
        self.assertTrue(np.allclose(np.triu(U), U))


class TestIntegerOverflowPolicy(unittest.TestCase):
    """Pin the documented integer-overflow policy.

    Per Philosophy.md / DType System.md / guides/release1/dtypes.md:
    - elementwise integer arithmetic wraps silently (C/NumPy two's-complement);
    - integer matmul reductions use a wider accumulator and raise OverflowError.
    """

    def test_elementwise_int32_add_wraps(self):
        r = pc.matrix([[2147483647]], dtype="int32") + pc.matrix([[1]], dtype="int32")
        self.assertEqual(_np(r).tolist(), [[-2147483648]])

    def test_elementwise_int32_mul_wraps(self):
        # 46341 * 46341 = 2147488281 -> wraps to -2147479015 in int32.
        r = pc.matrix([[46341]], dtype="int32") * pc.matrix([[46341]], dtype="int32")
        self.assertEqual(_np(r).tolist(), [[-2147479015]])

    def test_elementwise_uint8_add_wraps(self):
        r = pc.matrix([[255]], dtype="uint8") + pc.matrix([[1]], dtype="uint8")
        self.assertEqual(_np(r).tolist(), [[0]])

    def test_int32_matmul_overflow_raises(self):
        big = pc.matrix([[46341, 0], [0, 1]], dtype="int32")
        with self.assertRaises(OverflowError):
            big @ big


class TestPinv(unittest.TestCase):
    """Pseudoinverse baseline (normal equations) matches NumPy and satisfies A·P·A = A."""

    def test_pinv_tall_wide_square(self):
        rng = np.random.default_rng(0)
        for shape in [(5, 3), (3, 5), (4, 4)]:
            A = rng.standard_normal(shape)
            P = _np(pc.pinv(pc.matrix(A)))
            ref = np.linalg.pinv(A)
            self.assertTrue(np.allclose(P, ref, atol=1e-10), f"shape {shape}")
            self.assertTrue(np.allclose(A @ P @ A, A, atol=1e-10), f"shape {shape}")


if __name__ == "__main__":
    unittest.main()
