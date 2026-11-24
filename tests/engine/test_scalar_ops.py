"""Matrix ⊕ scalar operators vs NumPy: values AND result dtype (F-001, F-002).

Covers every dtype PyCauset exposes for dense matrices, the causal (triangular bit)
matrix, and triangular int32, against Python int/float/complex/bool scalars and
NumPy scalars. Documented divergence: bit/bool arithmetic counts into int32
(PyCauset policy, F-013), where NumPy would give int64.
"""
from __future__ import annotations

import operator

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float32", "float64", "complex64", "complex128", "bool"]

OPS = {
    "A+s": lambda a, s: a + s,
    "s+A": lambda a, s: s + a,
    "A-s": lambda a, s: a - s,
    "s-A": lambda a, s: s - a,
    "A*s": lambda a, s: a * s,
    "s*A": lambda a, s: s * a,
    "A/s": lambda a, s: a / s,
}

SCALARS = {
    "int3": 3,
    "float2.5": 2.5,
    "complex": 1 + 2j,
    "True": True,
    "np.int64": np.int64(3),
    "np.float32": np.float32(0.5),
}


def _operand(dtype: str) -> np.ndarray:
    base = np.array([[1, 2, 0], [3, 0, 4], [5, 6, 7]])
    return (base % 2).astype(bool) if dtype == "bool" else base.astype(dtype)


def _expected(op, a, s):
    with np.errstate(all="ignore"):
        if a.dtype == bool and not isinstance(s, (float, complex, np.floating, np.complexfloating)):
            # PyCauset: bit arithmetic counts into int32 (F-013).
            return op(a.astype(np.int32), s if not isinstance(s, np.integer) else s)
        return op(a, s)


@pytest.mark.parametrize("sname", list(SCALARS))
@pytest.mark.parametrize("opname", list(OPS))
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_op_matches_numpy(dtype, opname, sname):
    a, s, op = _operand(dtype), SCALARS[sname], OPS[opname]
    try:
        expected = np.asarray(_expected(op, a, s))
    except Exception:
        pytest.skip("NumPy itself rejects this combination")
    if expected.dtype.kind == "c" and a.dtype.kind in "ui" and np.issubdtype(type(s), np.unsignedinteger):
        pytest.skip("exotic")
    got = to_np(op(pc.matrix(a), s))
    np.testing.assert_allclose(got.astype(np.complex128), expected.astype(np.complex128), rtol=1e-6, atol=1e-6)
    assert got.dtype == expected.dtype, f"dtype {got.dtype} != NumPy {expected.dtype}"


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bool"])
def test_unary_minus_matches_numpy(dtype):
    a = _operand(dtype)
    with np.errstate(all="ignore"):
        expected = -a
    got = to_np(-pc.matrix(a))
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


def test_unary_minus_on_bit_raises_like_numpy():
    with pytest.raises((TypeError, ValueError)):
        -pc.matrix(np.eye(3, dtype=bool))


def test_python_int_out_of_dtype_range_raises_like_numpy():
    with pytest.raises(OverflowError):
        pc.matrix(np.ones((2, 2), dtype=np.int8)) * 1000


@pytest.mark.parametrize("opname", ["A*s", "s*A", "A/s"])
def test_causal_matrix_scaling_stays_triangular_and_exact(opname):
    c = pc.causal_matrix(30, populate=True)
    dense = to_np(c).astype(np.int64)
    out = OPS[opname](c, 3)
    expected = OPS[opname](dense, 3)
    np.testing.assert_allclose(to_np(out), expected)
    if opname != "A/s":
        assert type(out).__name__ == "TriangularIntegerMatrix"


def test_causal_matrix_plus_scalar_becomes_dense():
    c = pc.causal_matrix(10, populate=True)
    out = c + 2
    np.testing.assert_array_equal(to_np(out), to_np(c).astype(np.int32) + 2)
    assert "Triangular" not in type(out).__name__


def test_scalar_ops_on_disk_backed_integer_matrix(force_disk):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    m = pc.matrix(a)
    np.testing.assert_array_equal(to_np(m * 3), a * 3)
    np.testing.assert_array_equal(to_np(7 - m), 7 - a)


def test_scalar_result_is_independent_of_operand():
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    m = pc.matrix(a)
    r = m * 2
    r[0, 0] = 99
    assert m[0, 0] == 0
