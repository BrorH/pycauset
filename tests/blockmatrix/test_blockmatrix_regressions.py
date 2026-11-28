"""Block-matrix regressions from review 3.10 (review/areas/3.10-block-matrices.md)."""
import numpy as np
import pytest

import pycauset as pc
from pycauset._internal.blockmatrix import BlockMatrix
from pycauset._internal.thunks import StaleThunkError


def _grid(a):
    return pc.matrix([[pc.matrix(a)]])


def test_3_10_C1_block_results_feed_other_block_ops():
    """Review 3.10 C-1: ThunkBlock had no operators: (L @ R) @ R, (L + R) + R raised TypeError."""

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, -1.0], [2.0, 1.0]])
    L, R = _grid(a), _grid(b)
    cases = [
        (lambda: (L @ R) @ R, a @ b @ b), (lambda: L @ (R @ R), a @ (b @ b)),
        (lambda: (L + R) + R, a + b + b), (lambda: (L @ R) + R, a @ b + b),
        (lambda: pc.matmul(L @ R, R), a @ b @ b),
    ]
    for f, want in cases:
        np.testing.assert_allclose(np.asarray(f()), want)
    x = L @ R
    np.asarray(x)  # evaluated before reuse
    np.testing.assert_allclose(np.asarray(x @ R), a @ b @ b)


def test_3_10_C1_writes_two_operations_down_are_seen():

    a = pc.matrix(np.eye(2))
    L = pc.matrix([[a]])
    y = (L @ L) @ L
    np.asarray(y)
    a[0, 0] = 5.0
    with pytest.raises(StaleThunkError):
        np.asarray(y)


def test_3_10_C7_numpy_operands_use_the_block_operators():
    """Review 3.10 C-7: ndarray @ bm computed in NumPy; pc.add(bm, bm) densified."""

    a = np.arange(4.0).reshape(2, 2)
    L = _grid(a)
    for got, want in ((np.ones((2, 2)) @ L, np.ones((2, 2)) @ a), (np.ones((2, 2)) + L, 1 + a),
                      (pc.add(L, L), a + a), (pc.multiply(L, L), a * a)):
        assert isinstance(got, BlockMatrix), type(got)
        np.testing.assert_allclose(np.asarray(got), want)
