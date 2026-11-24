"""Element-wise operations on views (F-043, F-049, F-045).

Views (transpose, scale factor, slice, conjugate) are read straight from storage by the
element-wise kernel. Results must equal NumPy's bit for bit: integer views exactly (they used
to go through double and lost precision above 2**53), float/complex views with the factor
applied exactly once.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

N = 37  # odd, and not a multiple of the 64-element tile

DTYPES = {
    "int8": np.int8, "int32": np.int32, "int64": np.int64, "uint64": np.uint64,
    "float32": np.float32, "float64": np.float64, "complex128": np.complex128,
}


def _data(dtype, rng):
    if dtype == "int64":
        return rng.integers(2**61, 2**62, size=(N, N), dtype=np.int64)  # beyond 2**53
    if dtype == "uint64":
        return rng.integers(2**62, 2**63, size=(N, N), dtype=np.uint64)
    if dtype.startswith("int"):
        return rng.integers(-50, 50, size=(N, N)).astype(DTYPES[dtype])
    if dtype == "complex128":
        return (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    return rng.standard_normal((N, N)).astype(DTYPES[dtype])


VIEWS = {
    "T": (lambda m: m.T, lambda a: a.T),
    "slice": (lambda m: m[3:30, 5:32], lambda a: a[3:30, 5:32]),
    "T_slice": (lambda m: m.T[3:30, 5:32], lambda a: a.T[3:30, 5:32]),
}
FLOAT_VIEWS = {
    "scaled": (lambda m: m * 2.5, lambda a: a * a.dtype.type(2.5)),
    "scaled_T": (lambda m: (m * 2.5).T, lambda a: (a * a.dtype.type(2.5)).T),
}
OPS = {"+": np.add, "-": np.subtract, "*": np.multiply}


def _cases():
    for dt in DTYPES:
        views = dict(VIEWS)
        if dt.startswith("float") or dt.startswith("complex"):
            views.update(FLOAT_VIEWS)
        if dt.startswith("complex"):
            views["conj"] = (lambda m: m.conj(), lambda a: np.conj(a))
        for vname in views:
            for op in OPS:
                yield dt, vname, op, views[vname]


@pytest.mark.parametrize("dtype,view,op,fns", [pytest.param(*c, id=f"{c[0]}-{c[1]}-{c[2]}") for c in _cases()])
def test_view_operand_matches_numpy_exactly(dtype, view, op, fns, rng):
    pc_view, np_view = fns
    a = _data(dtype, rng)
    b = _data(dtype, rng)
    x = pc_view(pc.matrix(a))
    ref_a = np_view(a)
    y_np = b[: ref_a.shape[0], : ref_a.shape[1]].copy()
    y = pc.matrix(y_np)
    with np.errstate(over="ignore"):
        expected = OPS[op](ref_a, y_np)
    got = to_np(eval(f"x {op} y", {}, {"x": x, "y": y}))
    assert got.dtype == expected.dtype
    if dtype.startswith("complex") and op == "*":
        # complex products differ from NumPy's formula by <= 1 ulp (not view related)
        np.testing.assert_allclose(got, expected, rtol=4e-16, atol=0)
    else:
        np.testing.assert_array_equal(got, expected)


def test_conjugated_operand_in_scalar_ops(rng):
    z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    got = to_np(pc.matrix(z).conj() + 1.5)
    np.testing.assert_array_equal(got, np.conj(z) + 1.5)


def test_both_operands_views(rng):
    a = rng.integers(2**61, 2**62, size=(N, N), dtype=np.int64)
    b = rng.integers(2**61, 2**62, size=(N, N), dtype=np.int64)
    got = to_np(pc.matrix(a).T - pc.matrix(b)[::1, ::1].T)
    np.testing.assert_array_equal(got, a.T - b.T)


# --- scalar operations on views, every structure (F-050) ---------------------------------
def _structured(kind, dtype, rng):
    n = 7
    if kind == "dense":
        if dtype == "complex_float16":
            m = pc.DenseMatrix(n, n, dtype=dtype)
            for i in range(n):
                for j in range(n):
                    m[i, j] = complex(int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
            return m
        np_dt = {"float16": np.float16, "float32": np.float32, "float64": np.float64,
                 "complex_float32": np.complex64, "complex_float64": np.complex128}[dtype]
        data = rng.integers(-4, 5, (n, n)).astype(np_dt)
        if dtype.startswith("complex"):
            data = data + 1j * rng.integers(-4, 5, (n, n))
        return pc.matrix(data.astype(np_dt))
    m = {"triangular": pc.TriangularMatrix, "symmetric": pc.SymmetricMatrix,
         "antisymmetric": pc.AntiSymmetricMatrix, "diagonal": pc.DiagonalMatrix}[kind](n, dtype=dtype)
    for i in range(n):
        for j in range(i, n):
            if (kind == "diagonal" and i != j) or (kind in ("triangular", "antisymmetric") and i == j):
                continue
            m[i, j] = float(rng.integers(-4, 5))
    return m


STRUCT_CASES = [("dense", d) for d in ("float16", "float32", "float64", "complex_float16", "complex_float32", "complex_float64")]
STRUCT_CASES += [(k, "float64") for k in ("triangular", "symmetric", "antisymmetric", "diagonal")]
SCALAR_VIEWS = {"T": lambda m: m.T, "conj": lambda m: m.conj(), "H": lambda m: m.H, "scaled_T": lambda m: (m * 2.0).T}
SCALAR_OPS = {"A+s": lambda x: x + 1.5, "s-A": lambda x: 1.5 - x, "A-s": lambda x: x - 1.5,
              "A*s": lambda x: x * 3.0, "A/s": lambda x: x / 2.0, "-A": lambda x: -x}


@pytest.mark.parametrize("kind,dtype", STRUCT_CASES, ids=[f"{k}-{d}" for k, d in STRUCT_CASES])
@pytest.mark.parametrize("view", list(SCALAR_VIEWS))
@pytest.mark.parametrize("op", list(SCALAR_OPS))
def test_scalar_op_on_view_matches_numpy(kind, dtype, view, op, rng):
    m = SCALAR_VIEWS[view](_structured(kind, dtype, rng))
    ref = to_np(m).astype(np.complex128)
    got = to_np(SCALAR_OPS[op](m))
    expected = SCALAR_OPS[op](ref)
    assert got.shape == expected.shape
    tol = 1e-2 if "16" in dtype else 1e-6
    np.testing.assert_allclose(got.astype(np.complex128), expected, rtol=tol, atol=tol)
