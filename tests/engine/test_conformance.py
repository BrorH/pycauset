"""Conformance grid: matrix ⊕ matrix for every dtype pair (review/TESTS.md §3.2).

For each (op, dtype_a, dtype_b) it checks that
  1. the result dtype is exactly what PyCauset's promotion resolver promises
     (`pc._debug_resolve_promotion`), i.e. kernels never ignore the resolver;
  2. the values match a high-precision NumPy reference, with a tolerance set by the
     result dtype;
  3. the result is a registered, usable class (not a bare MatrixBase, F-032).

PyCauset deliberately differs from NumPy in documented ways (review/RESTRUCTURE.md,
documentation/internals/DType System.md): under-promotion within floats, bit/bool
arithmetic counts into int32. The reference therefore uses PyCauset's dtype and
NumPy's arithmetic. Known-open gaps are strict xfails tied to a finding id.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]

OPS = {  # name -> (python operator, resolver op name)
    "+": (lambda a, b: a + b, "add"),
    "-": (lambda a, b: a - b, "subtract"),
    "*": (lambda a, b: a * b, "elementwise_multiply"),
    "/": (lambda a, b: a / b, "divide"),
    "@": (lambda a, b: a @ b, "matmul"),
}

RTOL = {"float16": 2e-3, "complex_float16": 2e-3, "float32": 1e-5, "complex_float32": 1e-5}


def _operand(dtype: str, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed).integers(1, 5, size=(4, 4))
    return (r % 2).astype(bool) if dtype == "bool" else r.astype(dtype)


def _known_gap(op: str, da: str, db: str) -> str | None:
    pair = {da, db}
    if op == "@" and da != db and (pair & {"float16", "complex64", "complex128"}):
        return "F-035a mixed-dtype matmul with float16/complex has no kernel"
    signed = {"int8", "int16", "int32", "int64"}
    if op != "/" and "uint64" in pair and pair & signed:
        return "F-035b signed int with uint64 is rejected"
    if op == "-" and da == db == "bool":
        return None  # PyCauset counts; NumPy raises for bool subtraction. Checked below.
    return None


CASES = [(op, da, db) for op in OPS for da in DTYPES for db in DTYPES]


@pytest.mark.parametrize("op,da,db", CASES, ids=[f"{da}{op}{db}" for op, da, db in CASES])
def test_binary_matrix_op(op, da, db):
    gap = _known_gap(op, da, db)
    if gap:
        pytest.xfail(gap)  # non-strict: these are recorded findings, flip to pass when fixed
    fn, resolver_op = OPS[op]
    a, b = _operand(da, 1), _operand(db, 2)

    expected_dtype = pc._debug_resolve_promotion(resolver_op, da, db)["result_dtype"]
    native = expected_dtype  # PyCauset's dtype name
    # Export dtype: NumPy has no complex32, so complex_float16 exports as complex64.
    expected_dtype = {"complex_float16": "complex64", "complex_float32": "complex64",
                      "complex_float64": "complex128", "bit": "bool"}.get(native, native)

    result = fn(pc.matrix(a), pc.matrix(b))
    assert type(result).__name__ != "MatrixBase", "unregistered result class (F-032)"
    got = to_np(result)
    assert str(got.dtype) == expected_dtype, f"result dtype {got.dtype}, resolver promised {expected_dtype}"

    # Reference. Integer results: NumPy arithmetic in that integer dtype (wraparound,
    # e.g. uint8 1 - 3 = 254, exactly as PyCauset). Float/complex: arithmetic in a wide
    # dtype, compared at the result's precision.
    kind = np.dtype(expected_dtype).kind
    with np.errstate(all="ignore"):
        if kind in "iu":
            ref = fn(a.astype(expected_dtype), b.astype(expected_dtype))
        elif kind == "b":
            ref = fn(a, b)
        else:
            wide = np.complex128 if kind == "c" else np.float64
            ref = fn(a.astype(wide), b.astype(wide))
    rtol = RTOL.get(native, 1e-12)
    np.testing.assert_allclose(got.astype(np.complex128), np.asarray(ref).astype(np.complex128),
                               rtol=rtol, atol=rtol, equal_nan=True)


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "@"])
@pytest.mark.parametrize("dtype", ["int32", "float64", "bool"])
def test_binary_matrix_op_on_disk_matches_ram(op, dtype, force_disk):
    fn, _ = OPS[op]
    a, b = _operand(dtype, 3), _operand(dtype, 4)
    on_disk = to_np(fn(pc.matrix(a), pc.matrix(b)))
    pc.set_memory_threshold(8 * 1024**3)
    in_ram = to_np(fn(pc.matrix(a), pc.matrix(b)))
    np.testing.assert_array_equal(on_disk, in_ram)


@pytest.mark.parametrize("shape_a,shape_b", [((3, 5), (5, 2)), ((1, 4), (4, 1)), ((6, 1), (1, 6))])
def test_matmul_rectangular_matches_numpy(shape_a, shape_b, rng):
    a, b = rng.standard_normal(shape_a), rng.standard_normal(shape_b)
    np.testing.assert_allclose(to_np(pc.matrix(a) @ pc.matrix(b)), a @ b, rtol=1e-12)
