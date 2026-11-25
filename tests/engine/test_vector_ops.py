"""Vector arithmetic runs through the matrix machinery (math/VectorOps.hpp).

Oracle: a vector op must give exactly what the same op gives on the 1 x n matrix holding the
same values (dtype and values), and NumPy's values. Matrix dtype rules are tested elsewhere
(conformance grid); here we check that vectors inherit them, plus vector-only concerns
(orientation, bit padding, unit vectors, factors, errors).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]

OPS = {
    "v+2": lambda x: x + 2, "2-v": lambda x: 2 - x, "v-3": lambda x: x - 3, "v*3": lambda x: x * 3,
    "v/2": lambda x: x / 2, "2/v": lambda x: 2 / x, "v*2.5": lambda x: x * 2.5, "-v": lambda x: -x,
    "v+v": lambda x: x + x, "v-v": lambda x: x - x, "v*v": lambda x: x * x, "v/v": lambda x: x / x,
    "v+1j": lambda x: x + 1j,
}


def _values(dtype, n=7):
    if dtype == "bool":
        return np.arange(n) % 3 == 0
    return (np.arange(n) + 1).astype(dtype)


def _run(fn, obj):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(obj)
        except Exception as e:  # noqa: BLE001 - compared below
            return e


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", list(OPS))
def test_vector_op_equals_row_matrix_op(dtype, op):
    a = _values(dtype)
    got = _run(OPS[op], pc.vector(a))
    ref = _run(OPS[op], pc.matrix(a[None, :]))
    if isinstance(ref, Exception):
        assert type(got) is type(ref), (got, ref)
        return
    assert not isinstance(got, Exception), got
    assert isinstance(got, pc.DenseVector) and got.dtype == ref.dtype
    np.testing.assert_array_equal(to_np(got), to_np(ref)[0])
    with np.errstate(all="ignore"):
        expected = OPS[op](a.astype(np.int64) if dtype == "bool" and op != "v*v" else a)
    np.testing.assert_allclose(to_np(got).astype(complex), np.asarray(expected).astype(complex),
                               rtol=2e-3 if "16" in dtype else 1e-6, equal_nan=True)


def test_integers_are_exact_and_wrap_like_numpy():
    big = np.array([2**62 + 1, -(2**61) - 3], dtype=np.int64)
    np.testing.assert_array_equal(to_np(pc.vector(big) * 3), big * 3)
    np.testing.assert_array_equal(to_np(pc.vector(big) + pc.vector(big)), big + big)
    u = np.array([250, 3], dtype=np.uint8)
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(to_np(pc.vector(u) + 10), u + np.uint8(10))
        np.testing.assert_array_equal(to_np(-pc.vector(u)), -u)


def test_orientation_and_factor_carry_over():
    v = pc.vector(np.arange(5, dtype=np.int32))
    w = pc.vector(np.arange(5, dtype=np.int32))
    w.scalar = 3  # integer view: logical values 0, 3, 6, ...
    np.testing.assert_array_equal(to_np(w + 1), np.arange(5) * 3 + 1)
    np.testing.assert_array_equal(to_np(v * w), np.arange(5) ** 2 * 3)
    t = v.T + 1
    assert t.is_transposed() and to_np(t).ravel().tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("n", [1, 63, 64, 65, 200])
def test_bit_vectors_of_any_length(n):
    a = np.arange(n) % 3 == 1
    b = np.arange(n) % 2 == 0
    np.testing.assert_array_equal(to_np(pc.vector(a) * pc.vector(b)), (a & b).astype(to_np(pc.vector(a) * pc.vector(b)).dtype))
    np.testing.assert_array_equal(to_np(pc.vector(a) * 5), a.astype(np.int32) * 5)


def test_unit_vector_and_numpy_operands():
    e = pc.UnitVector(4, 2)
    np.testing.assert_array_equal(to_np(e * 2.0), [0, 0, 2, 0])
    v = pc.vector(np.ones(4))
    np.testing.assert_array_equal(to_np(v + np.arange(4.0)), 1 + np.arange(4.0))
    np.testing.assert_array_equal(to_np(np.arange(4.0) - v), np.arange(4.0) - 1)


def test_errors():
    v = pc.vector(np.ones(3))
    with pytest.raises(ValueError):
        v + pc.vector(np.ones(4))
    with pytest.raises(TypeError):
        v + "x"
    with pytest.raises(OverflowError):
        pc.vector(np.ones(3, dtype=np.int8)) + 1000
