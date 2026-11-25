"""The matmul rules (dispatch_matmul, review/PLAN.md Phase N.1): result structure and dtype.

Values for every dtype x structure x view are the atlas's job (tests/atlas); this file pins
what the atlas does not check: which structure each rule returns, and the overflow guards.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

REAL_TRI = ["float64", "float32", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


def _tri(n, dtype, diag, seed=0):
    rng = np.random.default_rng(seed)
    ref = np.triu(rng.integers(0, 3, (n, n)), 0 if diag else 1).astype(dtype)
    m = pc.TriangularMatrix(n, dtype=dtype, has_diagonal=diag)
    for i in range(n):
        for j in range(i if diag else i + 1, n):
            if ref[i, j]:
                m[i, j] = ref[i, j].item()
    return m, ref


@pytest.mark.parametrize("dtype", REAL_TRI)
@pytest.mark.parametrize("diags", [(True, True), (True, False), (False, False)])
def test_R3_upper_times_upper_stays_triangular(dtype, diags):
    a, ra = _tri(7, dtype, diags[0], 1)
    b, rb = _tri(7, dtype, diags[1], 2)
    c = a @ b
    assert isinstance(c, pc.TriangularMatrix)
    np.testing.assert_array_equal(to_np(c), ra @ rb)


def test_R3_mixed_dtypes_promote_and_stay_triangular():
    a, ra = _tri(6, "int8", True, 3)
    b, rb = _tri(6, "float32", True, 4)
    c = a @ b
    assert isinstance(c, pc.TriangularMatrix)
    np.testing.assert_allclose(to_np(c), ra.astype(np.float64) @ rb.astype(np.float64), rtol=1e-6)


def test_R3_lower_operand_goes_dense():
    a, ra = _tri(6, "float64", True, 5)
    b, rb = _tri(6, "float64", True, 6)
    c = a.T @ b
    assert not isinstance(c, pc.TriangularMatrix)
    np.testing.assert_allclose(to_np(c), ra.T @ rb)


def test_R3_integer_overflow_raises():
    a = pc.TriangularMatrix(3, dtype="int8", has_diagonal=True)
    a[0, 1] = 100
    a[1, 1] = 100
    with pytest.raises(OverflowError):
        a @ a


def test_R2_strict_bit_times_strict_bit_counts_paths():
    rng = np.random.default_rng(7)
    ref = np.triu(rng.integers(0, 2, (70, 70)), 1).astype(bool)
    c = pc.causal_matrix(ref)
    out = c @ c
    assert isinstance(out, pc.TriangularMatrix)
    np.testing.assert_array_equal(to_np(out), ref.astype(np.int32) @ ref.astype(np.int32))


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int16"])
def test_R2_bit_times_dense_keeps_dtype(dtype):
    rng = np.random.default_rng(8)
    ra = rng.integers(0, 2, (70, 65)).astype(bool)
    rb = rng.integers(-3, 4, (65, 9)).astype(dtype)
    c = pc.matrix(ra) @ pc.matrix(rb)
    assert c.dtype == dtype
    np.testing.assert_array_equal(to_np(c), ra.astype(dtype) @ rb)


@pytest.mark.parametrize("dtype", ["float64", "int32", "uint8", "int64"])
def test_R4_diag_times_diag_stays_diagonal(dtype):
    a = pc.DiagonalMatrix(5, dtype=dtype)
    b = pc.DiagonalMatrix(5, dtype=dtype)
    for i in range(5):
        a[i, i] = i + 1
        b[i, i] = 2
    c = a @ b
    assert isinstance(c, pc.DiagonalMatrix)
    np.testing.assert_array_equal(to_np(c), np.diag([2 * (i + 1) for i in range(5)]).astype(dtype))


def test_R4_integer_overflow_raises():
    a = pc.DiagonalMatrix(2, dtype="int8")
    a[0, 0] = 100
    with pytest.raises(OverflowError):
        a @ a


@pytest.mark.parametrize("dtype", ["float16", "complex_float16"])
def test_R5_half_precision_computes_wider_and_keeps_dtype(dtype):
    rng = np.random.default_rng(9)
    ra = rng.integers(-3, 4, (6, 6)).astype(np.float64)
    c = pc.matrix(ra, dtype=dtype) @ pc.matrix(ra, dtype=dtype)
    assert c.dtype == dtype
    np.testing.assert_allclose(to_np(c).real, ra @ ra)


def test_R5_complex_conjugate_view_is_normalised():
    rng = np.random.default_rng(10)
    ra = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    m = pc.matrix(ra)
    np.testing.assert_allclose(to_np(m.conj() @ m.H), ra.conj() @ ra.conj().T)
