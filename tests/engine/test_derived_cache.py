"""Derived-value cache (review/features/DERIVED_CACHE.md, D.1-D.2): whole-matrix scalar
results are kept on the storage and dropped by any write. A cached answer must always equal
what NumPy gives for the data as it is now."""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc

native = pc._native


def _hits():
    return native._derived_cache_stats()[0]


@pytest.fixture
def m():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 6)) + 6 * np.eye(6)
    return pc.matrix(a), a


def test_second_call_is_served_from_the_cache(m):
    mat, a = m
    first = pc.determinant(mat)
    before = _hits()
    second = pc.determinant(mat)
    assert _hits() == before + 1
    assert second == first
    assert second == pytest.approx(np.linalg.det(a))


@pytest.mark.parametrize("write", ["setitem", "set", "slice", "fill", "set_identity"])
def test_every_write_drops_the_cache(m, write):
    mat, a = m
    for f in (pc.determinant, pc.trace, pc.sum, pc.norm):
        f(mat)  # fill the cache
    v0 = native._storage_version(mat)
    if write == "setitem":
        mat[0, 0] = 100.0
        a = a.copy(); a[0, 0] = 100.0
    elif write == "set":
        mat.set(2, 3, -7.0)
        a = a.copy(); a[2, 3] = -7.0
    elif write == "slice":
        mat[1:3, 1:3] = np.ones((2, 2))
        a = a.copy(); a[1:3, 1:3] = 1.0
    elif write == "fill":
        mat.fill(2.0)
        a = np.full_like(a, 2.0)
    else:
        mat.set_identity()
        a = np.eye(6)
    assert native._storage_version(mat) != v0
    assert complex(pc.determinant(mat)) == pytest.approx(np.linalg.det(a))
    assert pc.trace(mat) == pytest.approx(np.trace(a))
    assert pc.sum(mat) == pytest.approx(a.sum())
    assert pc.norm(mat) == pytest.approx(np.linalg.norm(a))


def test_views_have_their_own_entries_and_windows_none(m):
    mat, a = m
    for view, ref in [(mat.T, a.T), (mat * 2.0, a * 2.0), (mat * -1.0, -a)]:
        assert complex(pc.determinant(view)) == pytest.approx(np.linalg.det(ref))
        assert complex(pc.determinant(view)) == pytest.approx(np.linalg.det(ref))  # warm
    w = mat[1:4, 1:4]
    before = _hits()
    pc.determinant(w)
    pc.determinant(w)
    assert _hits() == before  # windows are not cached
    assert complex(pc.determinant(w)) == pytest.approx(np.linalg.det(a[1:4, 1:4]))


def test_copy_on_write_keeps_each_side_right(m):
    mat, a = m
    t = mat.T  # shares the storage until a write
    assert pc.sum(t) == pytest.approx(a.sum())
    mat[0, 1] = 50.0  # mat copies; t keeps the old values
    a2 = a.copy(); a2[0, 1] = 50.0
    assert pc.sum(mat) == pytest.approx(a2.sum())
    assert pc.sum(t) == pytest.approx(a.sum())  # the cached value of the old storage still holds


def test_integer_and_vector_results(m):
    i = pc.matrix(np.arange(16, dtype=np.int64).reshape(4, 4))
    assert pc.sum(i) == 120 and pc.sum(i) == 120
    i[3, 3] = 0
    assert pc.sum(i) == 105
    v = pc.vector(np.arange(5.0))
    assert pc.sum(v) == 10.0 and pc.sum(v) == 10.0
    v[0] = 100.0
    assert pc.sum(v) == 110.0


def test_rank_and_norm_parameters_are_part_of_the_key(m):
    mat, a = m
    assert pc.norm(mat, "fro") == pytest.approx(np.linalg.norm(a, "fro"))
    assert pc.norm(mat, 1) == pytest.approx(np.linalg.norm(a, 1))
    assert pc.norm(mat, 1) == pytest.approx(np.linalg.norm(a, 1))
    assert pc.matrix_rank(mat) == np.linalg.matrix_rank(a)
    assert pc.matrix_rank(mat, tol=1e3) == np.linalg.matrix_rank(a, tol=1e3)


def test_closure_over_causal_bits_follows_its_source():
    c = pc.causal_matrix(np.triu(np.ones((5, 5), dtype=bool), 1))
    s = c + c.T
    assert pc.sum(s) == 20 and pc.sum(c) == 10
    c[0, 1] = False
    assert pc.sum(c) == 9
    assert pc.sum(s) == int(np.asarray(s).sum())


def test_off_switch():
    mat = pc.matrix(np.eye(3) * 2)
    native._set_derived_cache(False)
    try:
        before = _hits()
        pc.determinant(mat)
        pc.determinant(mat)
        assert _hits() == before
    finally:
        native._set_derived_cache(True)
