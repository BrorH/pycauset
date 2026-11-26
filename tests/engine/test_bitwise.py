"""Bitwise & | ^ ~ (math/Bitwise.cpp): NumPy's dtype rule, every structure, vectors, errors.

Values over every dtype x structure x view are the atlas's job; this file pins the dtype rule,
the vector path, the PyCauset extras (nand, nor, xnor) and the error types.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

PAIRS = [("bool", "bool"), ("bool", "int32"), ("int8", "uint8"), ("uint8", "uint16"), ("int32", "uint32"),
         ("int64", "int64"), ("uint64", "uint64")]


def _np(dtype):
    return np.dtype(bool) if dtype == "bool" else np.dtype(dtype)


@pytest.mark.parametrize("da,db", PAIRS)
@pytest.mark.parametrize("op", ["&", "|", "^"])
def test_dtype_rule_matches_numpy(da, db, op):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 8, (5, 6)).astype(_np(da))
    b = rng.integers(0, 8, (5, 6)).astype(_np(db))
    got = eval(f"x {op} y", {"x": pc.matrix(a), "y": pc.matrix(b)})
    want = eval(f"x {op} y", {"x": a, "y": b})
    assert got.dtype == ("bool" if want.dtype == bool else str(want.dtype))
    np.testing.assert_array_equal(to_np(got), want)


def test_structured_operands():
    rng = np.random.default_rng(2)
    ref = np.triu(rng.integers(0, 2, (70, 70)), 1).astype(bool)
    c = pc.causal_matrix(ref)
    d = pc.matrix(rng.integers(0, 2, (70, 70)).astype(bool))
    np.testing.assert_array_equal(to_np(c & d), ref & to_np(d))
    np.testing.assert_array_equal(to_np(c ^ c.T), ref ^ ref.T)
    np.testing.assert_array_equal(to_np(~c), ~ref)
    t = pc.TriangularMatrix(4, dtype="int16", has_diagonal=True)
    t[0, 0], t[1, 3] = 5, -3
    np.testing.assert_array_equal(to_np(~t), ~to_np(t))


@pytest.mark.parametrize("name,fn", [("nand", np.bitwise_and), ("nor", np.bitwise_or), ("xnor", np.bitwise_xor)])
def test_negated_extras(name, fn):
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.full((2, 2), 3, dtype=np.int8)
    got = getattr(pc, f"bitwise_{name}")(pc.matrix(a), pc.matrix(b))
    np.testing.assert_array_equal(to_np(got), ~fn(a, b))


def test_vectors():
    a = np.array([1, 0, 1, 1], dtype=bool)
    b = np.array([1, 1, 0, 1], dtype=bool)
    np.testing.assert_array_equal(to_np(pc.vector(a) & pc.vector(b)), a & b)
    np.testing.assert_array_equal(to_np(~pc.vector(a)), ~a)
    x = np.array([5, 6, 7], dtype=np.uint8)
    np.testing.assert_array_equal(to_np(pc.vector(x) ^ pc.vector(x[::-1].copy())), x ^ x[::-1])


def test_float_operands_raise_type_error():
    m = pc.matrix(np.ones((2, 2)))
    with pytest.raises(TypeError):
        m & m
    with pytest.raises(TypeError):
        ~m


def test_shape_mismatch_raises_value_error():
    a = pc.matrix(np.ones((2, 2), dtype=bool))
    b = pc.matrix(np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        a & b


def test_signed_with_uint64():
    """No integer holds both ranges. Bitwise: TypeError (as NumPy). Arithmetic: raises by
    decision F-035b (NumPy's float64 would round silently above 2^53)."""
    a = np.array([[1, -2]], dtype=np.int8)
    b = np.array([[3, 4]], dtype=np.uint64)
    with pytest.raises(TypeError):
        pc.matrix(a) & pc.matrix(b)
    with pytest.raises(ValueError):
        pc.matrix(a) + pc.matrix(b)
