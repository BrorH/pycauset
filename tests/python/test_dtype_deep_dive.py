"""Tests for the dtype deep-dive fixes.

Covers the `pc.bool` alias, bool/bit matmul promotion, `pc.dot` matrix
semantics, and the lazy dtype-deferred zeros/ones/empty allocation.
"""

import unittest

import numpy as np

import pycauset as pc


def _np(obj):
    return pc.to_numpy(obj)


def _causal_matrix(n=8, seed=1223):
    return pc.CausalSet(n, seed=seed).C


def _to_dense_bool(mat, n):
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            out[i, j] = 1 if mat.get(i, j) else 0
    return out


class TestBoolAlias(unittest.TestCase):
    def test_bool_alias_resolves(self):
        self.assertEqual(pc.bool, "bool")
        self.assertEqual(pc.bool, pc.bool_)

    def test_bool_alias_in_all(self):
        self.assertIn("bool", pc.__all__)

    def test_bool_alias_accepted_by_normalizer(self):
        from pycauset._internal.dtypes import normalize_dtype

        self.assertEqual(normalize_dtype(pc.bool, np_module=np), "bool")


class TestBoolBitMatmul(unittest.TestCase):
    def test_dense_bool_times_triangular_bit(self):
        n = 8
        C = _causal_matrix(n)
        B = pc.ones((n, n), dtype=pc.bool_)
        got = _np(B @ C)
        self.assertEqual(got.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(got, np.ones((n, n), dtype=int) @ _to_dense_bool(C, n))

    def test_triangular_bit_times_dense_bool(self):
        n = 8
        C = _causal_matrix(n)
        B = pc.ones((n, n), dtype=pc.bool_)
        got = _np(C @ B)
        self.assertEqual(got.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(got, _to_dense_bool(C, n) @ np.ones((n, n), dtype=int))

    def test_lazy_ones_times_bit_matches_explicit_bool(self):
        n = 8
        C = _causal_matrix(n)
        lazy = pc.ones((n, n))
        explicit = pc.ones((n, n), dtype=pc.bool_)
        np.testing.assert_array_equal(_np(lazy @ C), _np(explicit @ C))


class TestDotSemantics(unittest.TestCase):
    def test_dot_vector_vector_is_scalar(self):
        v1 = pc.vector([1.0, 2.0, 3.0])
        v2 = pc.vector([4.0, 5.0, 6.0])
        self.assertEqual(pc.dot(v1, v2), 32.0)

    def test_dot_matrix_matrix(self):
        a = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = pc.matrix(np.array([[5.0, 6.0], [7.0, 8.0]]))
        want = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(_np(pc.dot(a, b)), want)

    def test_dot_matrix_vector(self):
        m = pc.matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        v = pc.vector([1.0, 1.0, 1.0])
        want = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) @ np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(np.ravel(_np(pc.dot(m, v))), np.ravel(want))

    def test_dot_vector_matrix(self):
        m = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
        v = pc.vector([1.0, 1.0])
        want = np.array([1.0, 1.0]) @ np.array([[1.0, 2.0], [3.0, 4.0]])
        got = _np(pc.dot(v, m))
        self.assertEqual(got.shape, (2,))
        np.testing.assert_allclose(got, want)


class TestLazyAllocation(unittest.TestCase):
    def test_zeros_default_int32(self):
        got = _np(pc.zeros((2, 2)))
        self.assertEqual(got.dtype, np.dtype(np.int32))
        self.assertTrue(bool((got == 0).all()))

    def test_ones_default_int32(self):
        got = _np(pc.ones((2, 2)))
        self.assertEqual(got.dtype, np.dtype(np.int32))
        self.assertTrue(bool((got == 1).all()))

    def test_explicit_dtype_still_works(self):
        self.assertIsInstance(pc.ones((2, 2), dtype="float64"), pc.FloatMatrix)
        self.assertIsInstance(pc.ones((2, 2), dtype="bool"), pc.DenseBitMatrix)

    def test_empty_use_before_write_raises(self):
        with self.assertRaises(TypeError):
            pc.empty((2, 2)).get(0, 0)

    def test_empty_fill_deduces_int(self):
        e = pc.empty((2, 2))
        e.fill(5)
        self.assertEqual(e.dtype, "int32")

    def test_empty_fill_deduces_float(self):
        e = pc.empty((2, 2))
        e.fill(2.5)
        self.assertEqual(e.dtype, "float64")

    def test_empty_set_deduces_bool(self):
        e = pc.empty((2, 2))
        e.set(0, 0, True)
        self.assertEqual(e.dtype, "bool")

    def test_empty_set_deduces_complex(self):
        e = pc.empty((2, 2))
        e.set(0, 0, 1 + 2j)
        self.assertEqual(e.dtype, "complex_float64")

    def test_lazy_zeros_carries_zero_structure(self):
        self.assertTrue(pc.zeros((2, 2)).properties.get("is_zero"))

    def test_lazy_ones_carries_constant_structure(self):
        props = pc.ones((2, 2)).properties
        self.assertTrue(props.get("is_constant"))
        self.assertEqual(props.get("constant_value"), 1)

    def test_lazy_structures_recognized(self):
        from pycauset._internal.ops import _effective_structure_for

        self.assertEqual(_effective_structure_for(pc.zeros((2, 2))), "zero")
        self.assertEqual(_effective_structure_for(pc.ones((2, 2))), "constant")


class TestMixedDTypeMatmul(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.C = _causal_matrix(self.n)
        self.Cn = _to_dense_bool(self.C, self.n)

    def test_int32_times_float64(self):
        a = pc.ones((self.n, self.n), dtype="int32")
        b = pc.ones((self.n, self.n), dtype="float64")
        np.testing.assert_allclose(
            pc.to_numpy(a @ b), np.ones((self.n, self.n)) @ np.ones((self.n, self.n))
        )

    def test_float64_times_int32(self):
        a = pc.ones((self.n, self.n), dtype="float64")
        b = pc.ones((self.n, self.n), dtype="int32")
        np.testing.assert_allclose(
            pc.to_numpy(a @ b), np.ones((self.n, self.n)) @ np.ones((self.n, self.n))
        )

    def test_int32_times_triangular_bit(self):
        a = pc.ones((self.n, self.n), dtype="int32")
        np.testing.assert_allclose(
            pc.to_numpy(a @ self.C), np.ones((self.n, self.n), dtype=int) @ self.Cn
        )

    def test_triangular_bit_times_int32(self):
        b = pc.ones((self.n, self.n), dtype="int32")
        np.testing.assert_allclose(
            pc.to_numpy(self.C @ b), self.Cn @ np.ones((self.n, self.n), dtype=int)
        )

    def test_float64_times_triangular_bit(self):
        a = pc.ones((self.n, self.n), dtype="float64")
        np.testing.assert_allclose(
            pc.to_numpy(a @ self.C), np.ones((self.n, self.n)) @ self.Cn
        )

    def test_triangular_bit_times_float64(self):
        b = pc.ones((self.n, self.n), dtype="float64")
        np.testing.assert_allclose(
            pc.to_numpy(self.C @ b), self.Cn @ np.ones((self.n, self.n))
        )


class TestWideMixedIntMatmul(unittest.TestCase):
    """Mixed-width integer matmul promotes to the wider type (NumPy-compatible)."""

    def _check(self, da, db, na, nb):
        n = 12
        a = pc.ones((n, n), dtype=da)
        b = pc.ones((n, n), dtype=db)
        got = pc.to_numpy(a @ b)
        want = np.ones((n, n), dtype=na) @ np.ones((n, n), dtype=nb)
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(got, want)

    def test_int8_times_int16(self):
        self._check("int8", "int16", np.int8, np.int16)

    def test_int16_times_int8(self):
        self._check("int16", "int8", np.int16, np.int8)

    def test_int64_times_int32(self):
        self._check("int64", "int32", np.int64, np.int32)

    def test_int32_times_int64(self):
        self._check("int32", "int64", np.int32, np.int64)

    def test_uint8_times_uint16(self):
        self._check("uint8", "uint16", np.uint8, np.uint16)

    def test_uint32_times_int32_promotes_to_int64(self):
        self._check("uint32", "int32", np.uint32, np.int32)

    def test_uint64_times_uint32(self):
        self._check("uint64", "uint32", np.uint64, np.uint32)


if __name__ == "__main__":
    unittest.main()
