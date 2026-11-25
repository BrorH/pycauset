"""Linear algebra on every kind of view, against NumPy (F-059 ... F-063).

Kernels read storage directly; the entry points must hand them plain storage. Each view kind
(transpose, scale, conjugate, Hermitian, zero-offset window, offset window) is checked for
each operation and dtype.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["float32", "float64", "complex128", "int32"]
VIEWS = {
    "plain": (lambda m: m, lambda a: a),
    "T": (lambda m: m.T, lambda a: a.T),
    "scaled": (lambda m: m * 2, lambda a: a * 2),
    "conj": (lambda m: m.conj(), lambda a: a.conj()),
    "H": (lambda m: m.H, lambda a: a.conj().T),
    "window0": (lambda m: m[0:5, 0:5], lambda a: a[0:5, 0:5]),
    "window1": (lambda m: m[1:6, 0:5], lambda a: a[1:6, 0:5]),
}
B = np.linspace(-1, 1, 6)
OPS = {
    "inv": (lambda m, n: to_np(pc.invert(m)), lambda x, n: np.linalg.inv(x)),
    "solve": (lambda m, n: to_np(pc.solve(m, pc.vector(B[:n]))).ravel(), lambda x, n: np.linalg.solve(x, B[:n])),
    "eigvals": (lambda m, n: np.sort_complex(to_np(pc.eigvals(m)).astype(complex)),
                lambda x, n: np.sort_complex(np.linalg.eigvals(x).astype(complex))),
    "svdvals": (lambda m, n: to_np(pc.svdvals(m)), lambda x, n: np.linalg.svd(x, compute_uv=False)),
    "matmul": (lambda m, n: to_np(m @ m), lambda x, n: x @ x),
    "matrix_power": (lambda m, n: to_np(pc.matrix_power(m, 3)), lambda x, n: np.linalg.matrix_power(x, 3)),
    "pinv": (lambda m, n: to_np(pc.pinv(m)), lambda x, n: np.linalg.pinv(x)),
    "det": (lambda m, n: pc.determinant(m), lambda x, n: np.linalg.det(x)),
    "trace": (lambda m, n: pc.trace(m), lambda x, n: np.trace(x)),
    "export": (lambda m, n: np.asarray(m), lambda x, n: x),
}


def _matrix(dtype):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 6)) + (1j * rng.standard_normal((6, 6)) if "complex" in dtype else 0)
    a = a + 6 * np.eye(6)
    return np.round(a * 3).astype(dtype) if dtype == "int32" else a.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("op", list(OPS))
def test_linalg_on_views_matches_numpy(dtype, view, op):
    a = _matrix(dtype)
    m = VIEWS[view][0](pc.matrix(a))
    x = VIEWS[view][1](a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = OPS[op][0](m, x.shape[0])
    expected = OPS[op][1](x, x.shape[0])
    tol = 1e-3 if dtype == "float32" else 1e-7
    np.testing.assert_allclose(np.asarray(got, dtype=complex), np.asarray(expected, dtype=complex), rtol=tol, atol=tol)


def test_result_kinds_follow_numpy():
    z = pc.matrix(np.array([[1 + 2j, 3], [0, 2]]))
    assert pc.determinant(z) == pytest.approx(2 + 4j) and type(pc.determinant(z)) is complex
    assert type(pc.trace(z)) is complex
    i = pc.matrix(np.array([[2**62, 1], [0, 3]], dtype=np.int64))
    assert pc.trace(i) == 2**62 + 3 and type(pc.trace(i)) is int
    assert type(pc.determinant(pc.matrix(np.eye(2)))) is float


@pytest.mark.parametrize("dtype", ["float64", "float32", "int16", "int64", "uint8"])
def test_triangular_determinant_is_the_diagonal_product(dtype):
    t = pc.TriangularMatrix(3, dtype=dtype, has_diagonal=True)
    for i in range(3):
        t[i, i] = i + 2
    t[0, 2] = 1
    assert pc.determinant(t) == pytest.approx(24)


def test_rectangular_complex_pinv_and_singular_diagonal():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    np.testing.assert_allclose(to_np(pc.pinv(pc.matrix(r))), np.linalg.pinv(r), atol=1e-10)
    d = pc.DiagonalMatrix(3, dtype="float64")
    d[0, 0], d[1, 1], d[2, 2] = 1e-13, 2.0, 4.0
    np.testing.assert_allclose(np.diag(to_np(pc.invert(d))), [1e13, 0.5, 0.25])
