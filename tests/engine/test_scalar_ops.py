"""Matrix ⊕ scalar operators vs NumPy: values AND result dtype (F-001, F-002).

Covers every dtype PyCauset exposes for dense matrices, the causal (triangular bit)
matrix, and triangular int32, against Python int/float/complex/bool scalars and
NumPy scalars. Result dtypes follow DType System §9 (the matrix rules with the scalar as an
operand); values are NumPy's.
"""
from __future__ import annotations


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


_PC_NAME = {"complex64": "complex_float32", "complex128": "complex_float64"}
_NP_NAME = {"complex_float32": "complex64", "complex_float64": "complex128"}


def _scalar_dtype(matrix_dtype: str, s) -> str:
    """The dtype a scalar takes part as (DType System §9): a NumPy scalar its own; a Python
    scalar the matrix's within its kind (a float with an integer matrix: float64; a complex
    with a real matrix: the complex of its precision); a Python int with bool: int32; a bool
    (Python or NumPy) is bool (review 3.6 C-13)."""
    if isinstance(s, (bool, np.bool_)):
        return "bool"
    if isinstance(s, np.generic):
        return _PC_NAME.get(s.dtype.name, s.dtype.name)
    m = _PC_NAME.get(matrix_dtype, matrix_dtype)
    ints = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    if isinstance(s, int):
        return "int32" if m == "bool" else m
    if isinstance(s, float):
        return "float64" if m in ints else {"complex_float32": "float32", "complex_float64": "float64"}.get(m, m)
    return m if m.startswith("complex") else "complex_float64" if m in ints else {"float32": "complex_float32", "float64": "complex_float64", "float16": "complex_float16"}[m]


def _expected_dtype(opname: str, matrix_dtype: str, s) -> str | None:
    from tests.engine.test_dtype_rules import _expected

    op = {"+": "+", "-": "-", "*": "*", "/": "/"}[opname[1] if opname[0] == "A" else opname[1]]
    m = _PC_NAME.get(matrix_dtype, matrix_dtype)
    sd = _scalar_dtype(matrix_dtype, s)
    if m == "bool" and sd == "bool" and op == "*":
        return "bool"
    want = _expected(op, m, sd) if opname[0] == "A" else _expected(op, sd, m)
    return want


@pytest.mark.parametrize("sname", list(SCALARS))
@pytest.mark.parametrize("opname", list(OPS))
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_op_matches_numpy(dtype, opname, sname):
    """Values are NumPy's (computed in the result dtype); the result dtype is the documented
    rule (the matrix rules with the scalar as an operand, DType System §9)."""
    a, s, op = _operand(dtype), SCALARS[sname], OPS[opname]
    want = _expected_dtype(opname, dtype, s)
    if want is None:
        with pytest.raises((TypeError, ValueError)):
            op(pc.matrix(a), s)
        return
    with np.errstate(all="ignore"):
        wide = a.astype(np.int64) if dtype == "bool" and want != "bool" else a  # bit counting
        expected = np.asarray(op(wide.astype(np.complex128) if want.startswith("complex") else wide, s))
    got = to_np(op(pc.matrix(a), s))
    assert str(got.dtype) == _NP_NAME.get(want, want), f"dtype {got.dtype}, documented {want}"
    tol = 2e-3 if want in ("float16", "complex_float16") else 1e-6
    np.testing.assert_allclose(got.astype(np.complex128), expected.astype(np.complex128), rtol=tol, atol=tol)


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
        assert type(out) is pc.TriangularMatrix and out.dtype == "int32"


def test_causal_matrix_plus_scalar_becomes_dense():
    c = pc.causal_matrix(10, populate=True)
    out = c + 2
    np.testing.assert_array_equal(to_np(out), to_np(c).astype(np.int32) + 2)
    assert type(out) is pc.DenseMatrix


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
