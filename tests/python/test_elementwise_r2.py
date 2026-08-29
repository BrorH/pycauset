"""Extensive correctness suite for R2_CPU elementwise SIMD fast paths.

Covers the AVX2 f64 sub/mul/div kernels + the ``try_fast_simd`` wiring into
``CpuSolver::subtract/elementwise_multiply/elementwise_divide``, plus the
full-span guard that keeps strided submatrix views off the raw-pointer path.

Every test compares against NumPy (the alignment target per the NumPy
Alignment Protocol). See also `tests/BUG_LOG.md` for the view-contiguity bug
and the mixed-type null-deref that this suite regresses.
"""

import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pycauset

_STORAGE_TMP = tempfile.TemporaryDirectory()
pycauset.set_backing_dir(_STORAGE_TMP.name)


def _assert_close(test, got, ref, *, rtol=1e-12, atol=1e-12):
    """Convert a pycauset matrix to NumPy and compare, with a clear message."""
    got_np = np.asarray(got)
    test.assertEqual(got_np.shape, ref.shape, f"shape {got_np.shape} != {ref.shape}")
    np.testing.assert_allclose(got_np, ref, rtol=rtol, atol=atol)


# Sizes that straddle the SIMD block boundary (f64 block = 4096 elems,
# f32 block = 8192 elems) and odd tail sizes (non-multiples of the SIMD width).
_DENSE_SHAPES = [
    (1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (3, 5), (5, 3),
    (7, 7), (9, 9), (16, 16), (63, 63), (64, 64), (65, 65),
    (100, 7), (7, 100),
]


class TestElementwiseDenseF64(unittest.TestCase):
    """add/sub/mul/div on float64 dense matrices vs NumPy."""

    def _pair(self, shape):
        rng = np.random.default_rng(sum(shape) + 1)
        return rng.standard_normal(shape), rng.standard_normal(shape)

    def test_add_sub_mul_div_many_shapes(self):
        for shape in _DENSE_SHAPES:
            A, B = self._pair(shape)
            a, b = pycauset.matrix(A), pycauset.matrix(B)
            _assert_close(self, a + b, A + B)
            _assert_close(self, a - b, A - B)
            _assert_close(self, a * b, A * B)
            _assert_close(self, a / (b + 2.0), A / (B + 2.0))

    def test_against_full_value_grid(self):
        # Deterministic grid including negatives/zeros to exercise sign handling.
        A = np.array([[-3.0, 0.0, 2.5], [1e-12, -1.25, 4.0]])
        B = np.array([[1.0, 2.0, 0.0], [-0.5, 3.0, 1e12]])
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        _assert_close(self, a + b, A + B)
        _assert_close(self, a - b, A - B)
        _assert_close(self, a * b, A * B)
        _assert_close(self, a / (b + 2.0), A / (B + 2.0))

    def test_multiblock_path(self):
        # > 4096 elements forces the block-parallel path (multiple blocks).
        rng = np.random.default_rng(42)
        A = rng.standard_normal((128, 128))
        B = rng.standard_normal((128, 128))
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        _assert_close(self, a - b, A - B)
        _assert_close(self, a * b, A * B)
        _assert_close(self, a + b, A + B)


class TestElementwiseDenseF32(unittest.TestCase):
    """add/sub/mul/div on float32 dense matrices vs NumPy."""

    def test_add_sub_mul_div_many_shapes(self):
        rtol = atol = 1e-5
        for shape in _DENSE_SHAPES:
            rng = np.random.default_rng(sum(shape) + 2)
            A = rng.standard_normal(shape).astype(np.float32)
            B = rng.standard_normal(shape).astype(np.float32)
            a, b = pycauset.matrix(A), pycauset.matrix(B)
            _assert_close(self, a + b, A + B, rtol=rtol, atol=atol)
            _assert_close(self, a - b, A - B, rtol=rtol, atol=atol)
            _assert_close(self, a * b, A * B, rtol=rtol, atol=atol)
            _assert_close(self, a / (b + 2.0), A / (B + 2.0), rtol=rtol, atol=atol)

    def test_multiblock_path(self):
        # > 8192 elements forces the block-parallel path for float32.
        rng = np.random.default_rng(43)
        A = rng.standard_normal((128, 128)).astype(np.float32)
        B = rng.standard_normal((128, 128)).astype(np.float32)
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        _assert_close(self, a - b, A - B, rtol=1e-5, atol=1e-5)
        _assert_close(self, a * b, A * B, rtol=1e-5, atol=1e-5)


class TestElementwiseInt(unittest.TestCase):
    """Integer elementwise ops (add/sub/mul) match NumPy integer semantics."""

    def test_add_sub_mul(self):
        rng = np.random.default_rng(44)
        for shape in [(1, 1), (3, 5), (7, 7), (64, 64)]:
            A = rng.integers(-100, 100, size=shape).astype(np.int32)
            B = rng.integers(-100, 100, size=shape).astype(np.int32)
            a, b = pycauset.matrix(A), pycauset.matrix(B)
            _assert_close(self, a + b, A + B)
            _assert_close(self, a - b, A - B)
            _assert_close(self, a * b, A * B)


class TestElementwiseScalar(unittest.TestCase):
    """Scalar elementwise ops on full matrices and views."""

    def test_scalar_ops_full(self):
        rng = np.random.default_rng(45)
        A = rng.standard_normal((5, 5))
        a = pycauset.matrix(A)
        _assert_close(self, a * 3.0, A * 3.0)
        _assert_close(self, 3.0 * a, 3.0 * A)
        _assert_close(self, a + 2.0, A + 2.0)
        # `matrix - scalar` is not a supported operator; the equivalent is
        # addition of a negative scalar.
        _assert_close(self, a + (-2.0), A - 2.0)
        _assert_close(self, a / 4.0, A / 4.0)

    def test_scalar_ops_view(self):
        rng = np.random.default_rng(46)
        A = rng.standard_normal((6, 6))
        a = pycauset.matrix(A)
        av = a[1:4, 1:4]
        Ar = A[1:4, 1:4]
        _assert_close(self, av * 2.5, Ar * 2.5)
        _assert_close(self, av + 1.0, Ar + 1.0)
        _assert_close(self, av + (-1.0), Ar - 1.0)


class TestElementwiseViews(unittest.TestCase):
    """Regression: strided (incl. zero-offset) views must use element access."""

    def _slice_cases(self, parent):
        pr, pc = parent
        # (r0, r1, c0, c1), includes zero-offset (strided) and offset views.
        cases = [(0, 3, 0, 3), (0, 5, 2, 5), (1, 4, 1, 4)]
        if pr >= 6 and pc >= 6:
            cases += [(0, pr, 0, pc), (0, pr - 1, 0, pc - 1), (2, pr, 1, pc - 1)]
        return [(r0, r1, c0, c1) for (r0, r1, c0, c1) in cases
                if 0 <= r0 < r1 <= pr and 0 <= c0 < c1 <= pc]

    def test_view_elementwise_matches_numpy(self):
        for parent in [(5, 5), (6, 6), (8, 9), (20, 20)]:
            rng = np.random.default_rng(sum(parent) + 5)
            A = rng.standard_normal(parent)
            B = rng.standard_normal(parent)
            a, b = pycauset.matrix(A), pycauset.matrix(B)
            for (r0, r1, c0, c1) in self._slice_cases(parent):
                av, bv = a[r0:r1, c0:c1], b[r0:r1, c0:c1]
                Ar, Br = A[r0:r1, c0:c1], B[r0:r1, c0:c1]
                _assert_close(self, av + bv, Ar + Br)
                _assert_close(self, av - bv, Ar - Br)
                _assert_close(self, av * bv, Ar * Br)
                _assert_close(self, av / (bv + 2.0), Ar / (Br + 2.0))

    def test_eager_mul_view(self):
        # `__mul__` is eager and passes view operands straight to the solver -
        # the exact path that was wrong before the full-span fix.
        rng = np.random.default_rng(7)
        A = rng.standard_normal((5, 5))
        B = rng.standard_normal((5, 5))
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        av, bv = a[0:3, 0:3], b[0:3, 0:3]
        _assert_close(self, av * bv, A[0:3, 0:3] * B[0:3, 0:3])


class TestElementwiseMixedType(unittest.TestCase):
    """Mixed-dtype elementwise ops must not crash and must promote correctly."""

    def test_float64_plus_int32(self):
        f = pycauset.FloatMatrix(5)
        i = pycauset.IntegerMatrix(5)
        f[0, 1], i[0, 1] = 2.5, 3
        res = f + i
        self.assertAlmostEqual(res[0, 1], 5.5)

    def test_float64_minus_int32(self):
        f = pycauset.FloatMatrix(5)
        i = pycauset.IntegerMatrix(5)
        f[0, 1], i[0, 1] = 2.5, 3
        res = f - i
        self.assertAlmostEqual(res[0, 1], -0.5)

    def test_float64_times_int32(self):
        f = pycauset.FloatMatrix(5)
        i = pycauset.IntegerMatrix(5)
        f[0, 1], i[0, 1] = 2.5, 4
        res = f * i
        self.assertAlmostEqual(res[0, 1], 10.0)


class TestElementwiseBoundary(unittest.TestCase):
    """Empty, 1x1, zero-division, and in-place edge cases."""

    def test_empty_no_crash(self):
        a = pycauset.FloatMatrix(0)
        b = pycauset.FloatMatrix(0)
        self.assertEqual((a + b).size(), 0)
        self.assertEqual((a * b).size(), 0)

    def test_1x1(self):
        a = pycauset.matrix(np.array([[3.0]]))
        b = pycauset.matrix(np.array([[2.0]]))
        _assert_close(self, a * b, np.array([[6.0]]))
        _assert_close(self, a / b, np.array([[1.5]]))

    def test_zero_division_matches_numpy(self):
        A = np.array([[1.0, 0.0], [2.0, -0.0]])
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        got = np.asarray(a / b)
        ref = A / B  # inf/-inf/nan per IEEE
        np.testing.assert_array_equal(got, ref)

    def test_inplace_add_and_scalar_mul(self):
        A = pycauset.FloatMatrix(4)
        A[0, 1] = 10.0
        B = pycauset.FloatMatrix(4)
        B[0, 1] = 2.0
        A += B
        self.assertAlmostEqual(A[0, 1], 12.0)
        A *= 2.0
        self.assertAlmostEqual(A[0, 1], 24.0)


class TestElementwiseLarge(unittest.TestCase):
    """Large matrices exercise the multithreaded block-parallel SIMD path."""

    def test_large_f64(self):
        rng = np.random.default_rng(99)
        n = 512
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        a, b = pycauset.matrix(A), pycauset.matrix(B)
        _assert_close(self, a - b, A - B)
        _assert_close(self, a * b, A * B)


def tearDownModule():
    _STORAGE_TMP.cleanup()


if __name__ == "__main__":
    unittest.main()
