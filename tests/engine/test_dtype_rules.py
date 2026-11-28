"""Result dtypes of matrix (+ - * / @) matrix for every pair of real dtypes, against NumPy
with PyCauset's documented differences written out as data (documentation/internals/DType
System.md). Independent of the promotion resolver: test_conformance checks that kernels follow
the resolver, this checks that the resolver follows the rules (review 3.3 C-14).
Complex pairs wait for the complex rule to be decided and documented (3.3 C-7)."""
from __future__ import annotations

import operator
import warnings

import numpy as np
import pytest

import pycauset as pc

REAL = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64"]
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "@": operator.matmul}
_FLOAT_RANK = {"float16": 0, "float32": 1, "float64": 2}
_FLOATS = ["float16", "float32", "float64"]


def _float_for_int(dt: str) -> str:
    """The narrowest float that represents an integer (or bool) dtype (DType System §2.4)."""
    if dt in ("bool", "int8", "uint8"):
        return "float16"
    if dt in ("int16", "uint16"):
        return "float32"
    return "float64"


def _expected(op: str, a: str, b: str) -> str | None:
    """The documented result dtype; None where PyCauset raises by design."""
    fa, fb = a in _FLOATS, b in _FLOATS
    if fa and fb:  # §3: underpromotion within float
        return a if _FLOAT_RANK[a] <= _FLOAT_RANK[b] else b
    if fa or fb:  # §2.4: the float must hold the integer operand
        f, i = (a, b) if fa else (b, a)
        need = _float_for_int(i)
        return f if _FLOAT_RANK[f] >= _FLOAT_RANK[need] else need
    # Integers and bool.
    if op == "/":  # §2.5: at least float32; float64 when an operand is a 32/64-bit integer
        need = max((_float_for_int(a), _float_for_int(b), "float32"), key=_FLOAT_RANK.get)
        return need
    signed = {"int8", "int16", "int32", "int64"}
    if ("uint64" in (a, b)) and (a in signed or b in signed):
        return None  # §7, F-035b: NumPy's float64 would round an integer result
    if a == b == "bool":
        return "bool" if op == "*" else "int32"  # sums count (F-013); a product stays 0/1
    return np.result_type(a, b).name  # NumPy's integer rules


def _operand(dt: str):
    return pc.matrix(np.ones((2, 2), dtype=dt))


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("b", REAL)
@pytest.mark.parametrize("a", REAL)
def test_result_dtype_follows_the_documented_rules(op, a, b):
    want = _expected(op, a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if want is None:
            with pytest.raises((TypeError, ValueError)):
                OPS[op](_operand(a), _operand(b))
            return
        got = OPS[op](_operand(a), _operand(b))
    assert str(got.dtype) == want, f"{a} {op} {b}: {got.dtype}, documented {want}"
