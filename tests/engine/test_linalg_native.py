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


def _pc(dt) -> str:
    """PyCauset's name for a NumPy dtype."""
    name = str(np.dtype(dt))
    return {"complex64": "complex_float32", "complex128": "complex_float64", "bool": "bool"}.get(name, name)



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
    assert w.dtype == _pc(nw.dtype) and v.dtype == _pc(nv.dtype)
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
    assert w.dtype == _pc(nw.dtype) and v.dtype == _pc(nw.dtype)
    ww, vv = to_np(w), to_np(v)
    tol = 1e-3 if nw.dtype == np.complex64 else 1e-9
    np.testing.assert_allclose(a.astype(complex) @ vv, vv * ww, atol=tol * 10)
    got = to_np(pc.eigvals(pc.matrix(a)))
    assert max(np.min(np.abs(nw - g)) for g in got) < tol * 10  # same spectrum (order-free)


def test_eig_of_a_causal_matrix_is_nilpotent():
    c = pc.causal_matrix(np.triu(np.ones((5, 5), dtype=bool), 1))
    np.testing.assert_allclose(to_np(pc.eigvals(c)), np.zeros(5), atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cholesky(dtype):
    a = _spd(dtype)
    l = pc.cholesky(pc.matrix(a))
    nl = np.linalg.cholesky(a)
    assert l.dtype == _pc(nl.dtype)
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
    assert q.dtype == _pc(want[0].dtype)
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
    assert got.dtype == _pc(want.dtype)
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


# --- lu, outer, cross, vecdot, norm(ord), matrix_rank, cond -------------------------------

@pytest.mark.parametrize("dtype", ["int8", "float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_lu_reconstructs(dtype, shape):
    rng = np.random.default_rng(6)
    a = (rng.integers(-5, 6, shape) if dtype == "int8" else rng.standard_normal(shape)).astype(dtype)
    p, l, u = pc.lu(pc.matrix(a))
    k = min(shape)
    assert p.shape == (shape[0], shape[0]) and l.shape == (shape[0], k) and u.shape == (k, shape[1])
    pp, ll, uu = to_np(p), to_np(l), to_np(u)
    np.testing.assert_allclose(pp @ ll @ uu, a, atol=1e-4)
    np.testing.assert_allclose(np.tril(ll), ll)
    np.testing.assert_allclose(np.triu(uu), uu)


@pytest.mark.parametrize("da,db", [("int8", "int8"), ("bool", "bool"), ("int16", "float32"), ("float16", "float16"),
                                   ("complex64", "float64"), ("uint8", "int8")])
def test_outer_any_lengths_numpy_dtype(da, db):
    rng = np.random.default_rng(7)
    a = rng.integers(0, 100, 4).astype(da)
    b = rng.integers(0, 100, 6).astype(db)
    x, y = pc.vector(a), pc.vector(b)
    got = pc.outer(x, y)
    want = np.outer(a, b)
    # The engine's multiply dtype (NumPy's, except within-float under-promotion, F-034).
    assert got.dtype == pc._debug_resolve_promotion("elementwise_multiply", x.dtype, y.dtype)["result_dtype"]
    if want.dtype.kind in "biu":
        np.testing.assert_array_equal(to_np(got), want)  # integers wrap as NumPy's
    else:
        np.testing.assert_allclose(to_np(got), want, rtol=1e-3)


def test_outer_of_views():
    v = pc.vector(np.array([1 + 2j, 3j]))
    np.testing.assert_array_equal(to_np(pc.outer(v.conj(), v * 2)), np.outer(np.conj([1 + 2j, 3j]), np.array([1 + 2j, 3j]) * 2))


@pytest.mark.parametrize("dtype", ["int32", "float64", "complex128", "float16"])
def test_cross(dtype):
    a = np.array([1, 2, 3]).astype(dtype)
    b = np.array([4, -5, 6]).astype(dtype)
    got = pc.cross(pc.vector(a), pc.vector(b))
    want = np.cross(a, b)
    assert got.dtype == _pc(want.dtype)
    np.testing.assert_array_equal(to_np(got), want)


def test_cross_bool_raises():
    with pytest.raises(TypeError):  # NumPy: TypeError (boolean subtract)
        pc.cross(pc.vector(np.ones(3, dtype=bool)), pc.vector(np.ones(3, dtype=bool)))


def test_vecdot_conjugates_first():
    a, b = np.array([1 + 1j, 2]), np.array([3, 1j])
    assert complex(pc.vecdot(pc.vector(a), pc.vector(b))) == pytest.approx(np.vdot(a, b))


@pytest.mark.parametrize("ord", [None, "fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", ["int16", "float64", "complex128"])
def test_matrix_norms(ord, dtype):
    rng = np.random.default_rng(8)
    a = rng.integers(-9, 10, (4, 6)).astype(dtype)
    assert pc.norm(pc.matrix(a), ord) == pytest.approx(np.linalg.norm(a, ord), rel=1e-12)


@pytest.mark.parametrize("ord", [None, 0, 1, 2, 3, 0.5, np.inf, -np.inf])
def test_vector_norms(ord):
    x = np.array([3.0, -4.0, 0.0, 1.5])
    assert pc.norm(pc.vector(x), ord) == pytest.approx(np.linalg.norm(x, ord), rel=1e-12)


def test_matrix_rank_of_strict_triangle():
    """The old structural shortcut counted the non-zero diagonal: 0 here (F-090)."""
    ref = np.triu(np.ones((6, 6)), 1)
    assert pc.matrix_rank(pc.matrix(ref)) == np.linalg.matrix_rank(ref) == 5
    assert pc.matrix_rank(pc.causal_matrix(ref.astype(bool))) == 5
    assert pc.matrix_rank(pc.matrix(ref), tol=10.0) == np.linalg.matrix_rank(ref, tol=10.0)


@pytest.mark.parametrize("p", [None, 2, -2, 1, np.inf, "fro"])
def test_cond(p):
    a = np.array([[4.0, 1.0], [2.0, 3.0]])
    assert pc.cond(pc.matrix(a), p) == pytest.approx(np.linalg.cond(a, p), rel=1e-10)


@pytest.mark.parametrize("fn", [pc.eig, pc.eigvals, pc.eigh, pc.eigvalsh])
def test_eigen_of_non_square_raises(fn):
    with pytest.raises(ValueError, match="square"):
        fn(pc.matrix(np.ones((4, 3))))
