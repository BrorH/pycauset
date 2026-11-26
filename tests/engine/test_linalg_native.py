"""Eigen decompositions, Cholesky, QR and pinv (compute/cpu/CpuLinalg.cpp, F-090).

Values over every dtype x structure x view are the atlas's job; this file pins the result
dtypes (NumPy's LAPACK rule), the shapes of every QR mode, the error types, and that no
NumPy runs behind the call.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np



def _spd(dtype, n=6, seed=0):
    rng = np.random.default_rng(seed)
    if dtype.startswith("complex"):
        x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    else:
        x = rng.standard_normal((n, n))
    a = x @ x.conj().T + n * np.eye(n)
    return a.astype(dtype)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "complex64", "complex128"])
def test_eigh_dtype_and_values(dtype):
    a = _spd(dtype)
    w, v = pc.eigh(pc.matrix(a))
    nw, nv = np.linalg.eigh(a.astype(np.float32) if dtype == "float16" else a)
    assert w.dtype == str(nw.dtype) and v.dtype == str(nv.dtype)
    tol = 1e-3 if nw.dtype in (np.float32, np.complex64) else 1e-10
    np.testing.assert_allclose(to_np(w), nw, rtol=tol, atol=tol)
    vv = to_np(v)
    np.testing.assert_allclose(np.abs(vv.conj().T @ vv), np.eye(len(a)), atol=10 * tol)
    np.testing.assert_allclose(to_np(pc.eigvalsh(pc.matrix(a))), nw, rtol=tol, atol=tol)


def test_eigh_reads_the_lower_triangle():
    a = _spd("float64")
    junk = a.copy()
    junk[np.triu_indices(6, 1)] = 99.0
    np.testing.assert_allclose(to_np(pc.eigvalsh(pc.matrix(junk))), np.linalg.eigvalsh(junk))


@pytest.mark.parametrize("dtype", ["int32", "float32", "float64", "complex64", "complex128"])
def test_eig_dtype_and_residual(dtype):
    rng = np.random.default_rng(3)
    a = (rng.integers(-4, 5, (7, 7)) if dtype == "int32" else rng.standard_normal((7, 7))).astype(dtype)
    w, v = pc.eig(pc.matrix(a))
    nw = np.linalg.eigvals(a).astype(np.complex64 if dtype in ("float32", "complex64") else np.complex128)
    # Always complex: NumPy returns a real array when every eigenvalue happens to be real, a
    # dtype that depends on the values; PyCauset's depends on the input dtype only.
    assert w.dtype == str(nw.dtype) and v.dtype == str(nw.dtype)
    ww, vv = to_np(w), to_np(v)
    tol = 1e-3 if nw.dtype == np.complex64 else 1e-9
    np.testing.assert_allclose(a.astype(complex) @ vv, vv * ww, atol=tol * 10)
    np.testing.assert_allclose(np.sort_complex(to_np(pc.eigvals(pc.matrix(a)))), np.sort_complex(nw), atol=tol * 10)


def test_eig_of_a_causal_matrix_is_nilpotent():
    c = pc.causal_matrix(np.triu(np.ones((5, 5), dtype=bool), 1))
    np.testing.assert_allclose(to_np(pc.eigvals(c)), np.zeros(5), atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cholesky(dtype):
    a = _spd(dtype)
    l = pc.cholesky(pc.matrix(a))
    nl = np.linalg.cholesky(a)
    assert l.dtype == str(nl.dtype)
    np.testing.assert_allclose(to_np(l), nl, rtol=1e-4, atol=1e-4)


def test_cholesky_not_positive_definite_raises_value_error():
    with pytest.raises(ValueError):
        pc.cholesky(pc.matrix(-np.eye(3)))
    with pytest.raises(ValueError):
        pc.cholesky(pc.matrix(np.ones((2, 3))))


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
@pytest.mark.parametrize("dtype", ["float64", "complex128", "float32"])
def test_qr_shapes_and_reconstruction(shape, mode, dtype):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(shape).astype(dtype)
    if dtype == "complex128":
        a = a + 1j * rng.standard_normal(shape)
    out = pc.qr(pc.matrix(a), mode=mode)
    want = np.linalg.qr(a, mode=mode)
    if mode == "r":
        assert out.shape == want.shape
        np.testing.assert_allclose(np.abs(to_np(out)), np.abs(want), atol=1e-4)
        return
    q, r = out
    assert q.shape == want[0].shape and r.shape == want[1].shape
    assert q.dtype == str(want[0].dtype)
    np.testing.assert_allclose(to_np(q) @ to_np(r), a, atol=1e-4)
    np.testing.assert_allclose(np.triu(to_np(r)), to_np(r))


def test_qr_raw_mode_is_not_offered():
    with pytest.raises(ValueError):
        pc.qr(pc.matrix(np.eye(2)), mode="raw")


@pytest.mark.parametrize("dtype", ["float64", "float32", "complex128", "int16"])
@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_pinv(dtype, shape):
    rng = np.random.default_rng(5)
    a = (rng.integers(-3, 4, shape) if dtype == "int16" else rng.standard_normal(shape)).astype(dtype)
    if dtype == "complex128":
        a = a + 1j * rng.standard_normal(shape)
    got = pc.pinv(pc.matrix(a))
    want = np.linalg.pinv(a)
    assert got.dtype == str(want.dtype)
    np.testing.assert_allclose(to_np(got), want, rtol=1e-4, atol=1e-4)


def test_pinv_rank_deficient_matches_numpy():
    a = np.outer([1.0, 2.0, 3.0], [1.0, -1.0])
    np.testing.assert_allclose(to_np(pc.pinv(pc.matrix(a))), np.linalg.pinv(a), atol=1e-12)


@pytest.mark.parametrize("fn", [pc.eigh, pc.eigvalsh, pc.eig, pc.eigvals, pc.cholesky, pc.qr, pc.pinv])
def test_no_numpy_behind_the_call(fn, monkeypatch):
    a = pc.matrix(_spd("float64", n=4))
    for name in ("eig", "eigh", "eigvals", "eigvalsh", "cholesky", "qr", "pinv", "svd"):
        monkeypatch.setattr(np.linalg, name, lambda *x, **k: (_ for _ in ()).throw(AssertionError("NumPy ran")))
    fn(a)
