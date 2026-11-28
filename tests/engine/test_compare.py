"""Comparisons (review F-128, 3.6 C-7): element-wise bool results against NumPy."""
import operator

import numpy as np
import pytest

import pycauset as pc

OPS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
DTYPES = [np.bool_, np.int8, np.int64, np.uint8, np.uint64, np.float16, np.float32, np.float64,
          np.complex64, np.complex128]


def _vals(dt, seed):
    r = np.random.default_rng(seed)
    a = r.integers(-2, 3, (3, 4))
    if dt == np.bool_:
        return a > 0
    if np.issubdtype(dt, np.unsignedinteger):
        return np.abs(a).astype(dt)
    if np.issubdtype(dt, np.complexfloating):
        return (a + 1j * r.integers(-1, 2, (3, 4))).astype(dt)
    return a.astype(dt)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("da", DTYPES)
@pytest.mark.parametrize("db", [np.int64, np.uint64, np.float64, np.complex128, np.bool_])
def test_matrix_matrix(op, da, db):
    a, b = _vals(da, 1), _vals(db, 2)
    got = op(pc.matrix(a), pc.matrix(b))
    assert got.dtype == "bool"
    np.testing.assert_array_equal(np.asarray(got), op(a, b))


@pytest.mark.parametrize("op", OPS)
def test_views_structures_scalars_broadcasting(op):
    a = np.arange(9.0).reshape(3, 3) - 4
    A = pc.matrix(a)
    cases = [
        (A.T, a.T), (A * 2, a * 2), (A[0:3, 0:3], a),
        (pc.IdentityMatrix(3), np.eye(3)), (pc.causal_matrix(np.triu(np.ones((3, 3), bool), 1)), np.triu(np.ones((3, 3), bool), 1)),
        (pc.symmetric(a + a.T), a + a.T),
    ]
    for m, x in cases:
        for other, other_np in ((m, x), (0, 0), (1.5, 1.5), (np.int8(1), np.int8(1)), (x, x), (np.arange(3.0), np.arange(3.0)),
                                (pc.matrix(np.ones((1, 3))), np.ones((1, 3))), (pc.matrix(np.ones((3, 1))), np.ones((3, 1)))):
            np.testing.assert_array_equal(np.asarray(op(m, other)), op(x, other_np))
    w = A[0:2, 1:3]
    np.testing.assert_array_equal(np.asarray(op(w, w)), op(a[0:2, 1:3], a[0:2, 1:3]))
    # reflected: 0 < A is A > 0
    np.testing.assert_array_equal(np.asarray(op(0.5, A)), op(0.5, a))


def test_exact_integers_and_nan():
    big = np.array([[2**63 - 1, -1]], np.int64)
    ubig = np.array([[2**63, 2**64 - 1]], np.uint64)
    for op in OPS:
        np.testing.assert_array_equal(np.asarray(op(pc.matrix(big), pc.matrix(ubig))), op(big, ubig))
    n = np.array([[np.nan, 1.0]])
    for op in OPS:
        np.testing.assert_array_equal(np.asarray(op(pc.matrix(n), pc.matrix(n))), op(n, n))
    i8 = np.array([[1, -5]], np.int8)
    np.testing.assert_array_equal(np.asarray(pc.matrix(i8) < 300), i8 < 300)  # NumPy 2: by value


def test_vectors():
    x, y = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    for op in OPS:
        got = op(pc.vector(x), pc.vector(y))
        assert isinstance(got, pc.VectorBase)
        np.testing.assert_array_equal(np.asarray(got), op(x, y))
        np.testing.assert_array_equal(np.asarray(op(pc.vector(x), 2.0)), op(x, 2.0))


def test_shape_mismatch_and_unhashable():
    with pytest.raises(ValueError, match="broadcast"):
        pc.matrix(np.ones((2, 3))) == pc.matrix(np.ones((3, 2)))
    assert (pc.matrix(np.eye(2)) == "a") is False  # NotImplemented both ways: identity, as NumPy's object fallback
    with pytest.raises(TypeError):
        hash(pc.matrix(np.eye(2)))


def test_truth_value_as_numpy():
    """bool(M) was always True, so `if A == B:` passed for unequal matrices."""
    assert bool(pc.matrix(np.array([[2.0]]))) and not bool(pc.matrix(np.array([[0.0]])))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(pc.matrix(np.eye(2)) == pc.matrix(np.eye(2)))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(pc.vector([1.0, 2.0]))
    assert bool(pc.vector([3.0]))
