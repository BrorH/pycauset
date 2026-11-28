"""Operations with no RAM left: every result is file-backed and every kernel runs on mapped
storage (out of core is the mapping; the OS pages). Values against NumPy.

A huge safety margin makes the memory governor see no free RAM, so every new object goes to
a file. (The tiled StreamingManager that ran in this case is gone: it was slower than paging
and over-allocated, review 3.5; broadcasting raised there, C-3.)
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

native = pc._native


class _NoRam:
    """Context manager: inside it, nothing fits in RAM."""

    def __enter__(self):
        self.previous = native._get_ram_safety_margin()
        native._set_ram_safety_margin(1 << 62)
        return self

    def __exit__(self, *exc):
        native._set_ram_safety_margin(self.previous)
        return False


@pytest.fixture
def no_ram():
    return _NoRam()


def _file_backed(m) -> bool:
    return m.get_backing_file() not in ("", ":memory:")


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64"])
@pytest.mark.parametrize("shape", [(64, 64), (97, 53)])
@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_elementwise_out_of_core_matches_numpy(op, shape, dtype, no_ram, rng):
    a = (rng.integers(1, 9, size=shape)).astype(dtype)
    b = (rng.integers(1, 9, size=shape)).astype(dtype)
    fn = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide}[op]
    x, y = pc.matrix(a), pc.matrix(b)  # operands are created with the normal budget
    with no_ram:
        result = eval(f"x {op} y", {}, {"x": x, "y": y})
    assert _file_backed(result)
    got = to_np(result)
    expected = fn(a.astype(np.complex128), b.astype(np.complex128))
    np.testing.assert_allclose(got.astype(np.complex128), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("b", [np.arange(1.0, 31), np.arange(1.0, 41)[:, None], np.arange(1.0, 31)[None, :]],
                         ids=["vector", "column", "row"])
@pytest.mark.parametrize("op", ["-", "*"])
def test_broadcasting_out_of_core(op, b, no_ram, rng):
    """3.5 C-3: `M - row` / `M * col` raised under the streamed path."""
    a = rng.standard_normal((40, 30))
    x = pc.matrix(a)
    y = pc.vector(b) if b.ndim == 1 else pc.matrix(b)
    with no_ram:
        result = x - y if op == "-" else x * y
    np.testing.assert_allclose(to_np(result), a - b if op == "-" else a * b)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("m,k,n", [(64, 64, 64), (130, 70, 90), (1, 200, 1), (300, 5, 300)])
def test_matmul_out_of_core_matches_numpy(m, k, n, dtype, no_ram, rng):
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    x, y = pc.matrix(a), pc.matrix(b)
    with no_ram:
        result = x @ y
    got = to_np(result)
    tol = 1e-4 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(got, a.astype(np.float64) @ b.astype(np.float64), rtol=tol, atol=tol * k)


VIEW_CASES = {
    "A.T @ B": (lambda x, y: x.T @ y, lambda a, b: a.T @ b),
    "(A*2.5) @ B": (lambda x, y: (x * 2.5) @ y, lambda a, b: (a * 2.5) @ b),
    "A @ (B*-1.5)": (lambda x, y: x @ (y * -1.5), lambda a, b: a @ (b * -1.5)),
    "A.T + B": (lambda x, y: x.T + y, lambda a, b: a.T + b),
    "(A*2.5) - B": (lambda x, y: (x * 2.5) - y, lambda a, b: (a * 2.5) - b),
}


@pytest.mark.parametrize("case", list(VIEW_CASES))
def test_views_out_of_core(case, no_ram, rng):
    pc_fn, np_fn = VIEW_CASES[case]
    a = rng.standard_normal((70, 70))
    b = rng.standard_normal((70, 70))
    x, y = pc.matrix(a), pc.matrix(b)
    with no_ram:
        result = pc_fn(x, y)
    np.testing.assert_allclose(to_np(result), np_fn(a, b), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("dtypes", [("complex128", "complex128"), ("float64", "int32"), ("float32", "float64")])
@pytest.mark.parametrize("op", ["+", "*", "@"])
def test_mixed_dtypes_out_of_core(op, dtypes, no_ram, rng):
    a = rng.integers(1, 5, size=(40, 40)).astype(dtypes[0])
    b = rng.integers(1, 5, size=(40, 40)).astype(dtypes[1])
    x, y = pc.matrix(a), pc.matrix(b)
    with no_ram:
        result = eval(f"x {op} y", {}, {"x": x, "y": y})
    expected = eval(f"a {op} b", {}, {"a": a.astype(np.complex128), "b": b.astype(np.complex128)})
    np.testing.assert_allclose(to_np(result).astype(np.complex128), expected, rtol=1e-5, atol=1e-5)
