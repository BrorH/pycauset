"""Every consumer reads a scaled integer matrix correctly (F-042 / LAZY.md L1).

`k * A` on an integer matrix may be an O(1) view (factor stored, storage shared). The failure
this guards against is F-001: readers that ignore the factor. Each consumer below is compared
with NumPy on the same computation, exactly (integer arithmetic wraps like NumPy's).
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

N = 9
DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint32", "uint64"]
FACTORS = [3, -2, 0, 127]


def _pair(dtype, rng):
    info = np.iinfo(dtype)
    lo, hi = max(info.min, -100), min(info.max, 100)
    a = rng.integers(lo, hi, size=(N, N)).astype(dtype)
    if dtype == "int64":
        a[0, 0] = 2**62 + 12345  # beyond 2**53: must stay exact
    return a


def _np_scale(a, k):
    """k * a with NumPy's wrapping integer semantics (computed mod 2**64, then narrowed)."""
    with np.errstate(over="ignore"):
        return (a.astype(np.uint64) * np.uint64(k % 2**64)).astype(a.dtype)


CONSUMERS = {
    "getitem": (lambda m: np.array([[m[i, j] for j in range(N)] for i in range(N)]), lambda a: a),
    "get": (lambda m: np.array([[m.get(i, j) for j in range(N)] for i in range(N)]), lambda a: a),
    "to_numpy": (lambda m: to_np(m), lambda a: a),
    "T": (lambda m: to_np(m.T), lambda a: a.T),
    "slice": (lambda m: to_np(m[2:7, 1:8]), lambda a: a[2:7, 1:8]),
    "neg": (lambda m: to_np(-m), lambda a: (np.zeros_like(a) - a)),
    "times_again": (lambda m: to_np(m * 5), lambda a: _np_scale(a, 5)),
    "plus_matrix": (lambda m: to_np(m + pc.matrix(np.ones((N, N), dtype=to_np(m).dtype))),
                    lambda a: a + np.ones_like(a)),
    "T_plus_matrix": (lambda m: to_np(m.T + pc.matrix(np.ones((N, N), dtype=to_np(m).dtype))),
                      lambda a: a.T + np.ones_like(a)),
    "plus_scalar": (lambda m: to_np(m + 1), lambda a: a + a.dtype.type(1)),
    "matmul": (lambda m: to_np(m @ pc.matrix(np.eye(N, dtype=to_np(m).dtype))), lambda a: a),
    "matvec": (lambda m: to_np(m @ pc.vector(np.ones(N, dtype=to_np(m).dtype))),
               lambda a: a.sum(axis=1, dtype=a.dtype)),
    "vecmat": (lambda m: to_np(pc.vector(np.ones(N, dtype=to_np(m).dtype)) @ m).ravel(),  # row vector
               lambda a: a.sum(axis=0, dtype=a.dtype)),
    "sum": (lambda m: np.asarray(pc.sum(m)), lambda a: np.asarray(a.astype(np.float64).sum())),
    "invert": (lambda m: to_np(~m), lambda a: ~a),
}


def _save_load(m, tmp_path):
    p = tmp_path / "v.pycauset"
    pc.save(m, p)
    return to_np(pc.load(p))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", FACTORS)
@pytest.mark.parametrize("consumer", list(CONSUMERS))
def test_scaled_integer_matrix_consumers(dtype, k, consumer, rng):
    if consumer == "neg" and dtype.startswith("uint"):
        pytest.skip("unsigned negation follows NumPy's wrap; covered in test_scalar_ops")
    a = _pair(dtype, rng)
    if k < 0 and dtype.startswith("uint"):
        with pytest.raises(OverflowError):  # NumPy 2: negative Python int vs unsigned array
            pc.matrix(a) * k
        return
    m = pc.matrix(a) * k
    ref = _np_scale(a, k)
    assert m.dtype == dtype
    pc_fn, np_fn = CONSUMERS[consumer]
    with np.errstate(over="ignore"):
        expected = np_fn(ref)
    got = pc_fn(m)
    if consumer == "sum":
        np.testing.assert_allclose(complex(got).real, float(expected), rtol=1e-12)
    else:
        np.testing.assert_array_equal(np.asarray(got).astype(expected.dtype), expected)


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_scaled_integer_matrix_save_load(dtype, rng, tmp_path):
    a = _pair(dtype, rng)
    m = pc.matrix(a) * 3
    np.testing.assert_array_equal(_save_load(m, tmp_path), _np_scale(a, 3))


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_chained_scaling_folds_exactly(dtype, rng):
    a = _pair(dtype, rng)
    m = 2 * (3 * (5 * pc.matrix(a)))
    np.testing.assert_array_equal(to_np(m), _np_scale(_np_scale(_np_scale(a, 5), 3), 2))


def test_source_unchanged_after_writing_into_scaled_view(rng):
    a = _pair("int32", rng)
    src = pc.matrix(a)
    v = src * 3
    v[0, 1] = 7
    assert v[0, 1] == 7 and src[0, 1] == a[0, 1]
    np.testing.assert_array_equal(to_np(src), a)


def test_integer_scaling_is_a_view():
    a = pc.matrix(np.arange(16, dtype=np.int32).reshape(4, 4))
    v = 5 * (3 * (2 * a))
    assert v.scalar == 30 and v.dtype == "int32"  # O(1): one folded factor, shared storage
    np.testing.assert_array_equal(to_np(v), np.arange(16, dtype=np.int32).reshape(4, 4) * 30)


@pytest.mark.parametrize("dtype", ["int16", "int64"])
def test_triangular_integer_views(dtype):
    t = pc.TriangularMatrix(6, dtype=dtype)
    t[0, 3] = 7
    t[1, 5] = -4
    tn = to_np(t)
    v = t * -5
    assert type(v) is pc.TriangularMatrix and v.dtype == dtype
    np.testing.assert_array_equal(to_np(v), tn * -5)
    np.testing.assert_array_equal(to_np(v + 1), tn * -5 + 1)
    np.testing.assert_array_equal(to_np(v @ t), (tn * -5) @ tn)
    np.testing.assert_array_equal(to_np(v.T), (tn * -5).T)


def test_mixed_consumers_of_integer_views(rng):
    a = rng.integers(-9, 9, (6, 6)).astype(np.int32)
    f = rng.standard_normal((6, 6))
    v, ref = pc.matrix(a) * 3, a * 3
    np.testing.assert_allclose(to_np(v @ pc.matrix(f)), ref @ f)
    np.testing.assert_allclose(to_np(v + pc.matrix(f)), ref + f)
    np.testing.assert_allclose(to_np(v * 2.5), ref * 2.5)
    np.testing.assert_allclose(to_np(v / 2), ref / 2)
    assert pc.trace(v) == np.trace(ref)
    np.testing.assert_allclose(pc.norm(v), np.linalg.norm(ref))
    c = pc.causal_matrix(6, seed=2)
    np.testing.assert_array_equal(to_np(c @ v), to_np(c).astype(np.int32) @ ref)
