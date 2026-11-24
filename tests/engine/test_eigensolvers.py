"""Iterative eigensolvers needed for R1: eigvals_arnoldi, eigvals_skew, eig_skew (F-030).

Reference: NumPy's dense eigensolver. Complex spectra are compared as multisets via
(|lambda|, Im lambda) ordering, because real parts of +-i*lambda pairs are only zero up
to rounding noise and would scramble a lexicographic complex sort.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np


def _top_k(vals, k):
    vals = np.asarray(vals)
    return vals[np.argsort(-np.abs(vals), kind="stable")][:k]


def _canon(vals):
    vals = np.asarray(vals, dtype=np.complex128)
    order = np.lexsort((np.round(vals.imag, 8), np.round(np.abs(vals), 8)))
    return vals[order]


def _assert_same_spectrum(got, ref, rtol=1e-9):
    got, ref = _canon(got), _canon(ref)
    scale = np.abs(ref).max()
    np.testing.assert_allclose(got, ref, rtol=0, atol=rtol * scale)


def _causal_dense(n, seed=1):
    return to_np(pc.causal_matrix(n, populate=True, seed=seed)).astype(np.float64)


# ------------------------------------------------------------------ eigvals_arnoldi
@pytest.mark.parametrize("n,k,m", [(60, 4, 20), (300, 6, 40)])
def test_arnoldi_symmetric_matches_numpy_and_is_real(n, k, m, rng):
    a = rng.standard_normal((n, n))
    a = a + a.T
    w = to_np(pc.eigvals_arnoldi(pc.matrix(a), k, m, 1e-10))
    assert w.dtype == np.float64
    _assert_same_spectrum(w, _top_k(np.linalg.eigvalsh(a), k))


def test_arnoldi_nonsymmetric_matches_numpy(rng):
    a = rng.standard_normal((200, 200))
    w = to_np(pc.eigvals_arnoldi(pc.matrix(a), 6, 40, 1e-10))
    assert w.dtype == np.complex128
    _assert_same_spectrum(w, _top_k(np.linalg.eigvals(a), 6))


@pytest.mark.parametrize("kind", ["C+C.T", "C-C.T"])
def test_arnoldi_on_causal_matrices(kind):
    c = _causal_dense(200)
    a = c + c.T if kind == "C+C.T" else c - c.T
    w = to_np(pc.eigvals_arnoldi(pc.matrix(a), 6, 40, 1e-10))
    _assert_same_spectrum(w, _top_k(np.linalg.eigvals(a), 6))


def test_arnoldi_disk_backed(force_disk, rng):
    a = rng.standard_normal((150, 150))
    a = a + a.T
    m = pc.matrix(a)
    assert m.get_backing_file() != ":memory:"
    _assert_same_spectrum(to_np(pc.eigvals_arnoldi(m, 5, 30, 1e-10)), _top_k(np.linalg.eigvalsh(a), 5))


def test_arnoldi_is_deterministic(rng):
    a = rng.standard_normal((120, 120))
    m = pc.matrix(a)
    first = to_np(pc.eigvals_arnoldi(m, 5, 30, 1e-10))
    second = to_np(pc.eigvals_arnoldi(m, 5, 30, 1e-10))
    np.testing.assert_array_equal(first, second)


def test_arnoldi_small_problem_uses_exact_fallback(rng):
    a = rng.standard_normal((5, 5))
    _assert_same_spectrum(to_np(pc.eigvals_arnoldi(pc.matrix(a), 4, 20, 1e-10)), _top_k(np.linalg.eigvals(a), 4))


def test_arnoldi_clustered_spectrum_is_accurate(rng):
    # Top eigenvalues 10, 10 - 1e-6, 10 - 2e-6: the unrestarted solver this replaced
    # returned values off by ~1e-3 on far easier inputs (F-030).
    q, _ = np.linalg.qr(rng.standard_normal((150, 150)))
    lam = np.concatenate([[10.0, 10.0 - 1e-6, 10.0 - 2e-6], rng.uniform(-5, 5, 147)])
    a = (q * lam) @ q.T
    a = (a + a.T) / 2
    w = to_np(pc.eigvals_arnoldi(pc.matrix(a), 3, 40, 1e-12))
    np.testing.assert_allclose(np.sort(w), np.sort(lam[:3]), rtol=0, atol=1e-9)


# ------------------------------------------------------------------ skew solvers
@pytest.mark.parametrize("source", ["random", "causal"])
@pytest.mark.parametrize("n,k", [(40, 6), (121, 10)])
def test_eigvals_skew_matches_numpy(source, n, k, rng):
    if source == "causal":
        c = _causal_dense(n)
        a = c - c.T
    else:
        r = rng.standard_normal((n, n))
        a = r - r.T
    w = to_np(pc.eigvals_skew(pc.matrix(a), k))
    _assert_same_spectrum(w, _top_k(np.linalg.eigvals(a), k), rtol=1e-8)


def test_eig_skew_vectors_satisfy_the_eigen_equation(rng):
    r = rng.standard_normal((50, 50))
    a = r - r.T
    w, v = pc.eig_skew(pc.matrix(a), 8)
    w, v = to_np(w), to_np(v)
    for j in range(8):
        resid = np.linalg.norm(a @ v[:, j] - w[j] * v[:, j])
        assert resid <= 1e-8 * np.abs(w).max()
