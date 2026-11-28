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


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_3_10_C2_elementwise_with_different_partitions(op):
    """Review 3.10 C-2: grids with different partitions failed in 7 of 8 cases; R + L gave Python
    tiles that exported as an object array."""
    f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
         "div": lambda x, y: x / y}[op]
    a = np.arange(1.0, 17.0).reshape(4, 4)
    c = np.full((4, 4), 2.0)
    A, C = pc.matrix(a), pc.matrix(c)
    L = pc.matrix([[A[:, :2], A[:, 2:]]])
    R = pc.matrix([[C]])
    for got, want in ((f(L, R), f(a, c)), (f(R, L), f(c, a))):
        arr = np.asarray(got)
        assert arr.dtype == np.float64 and arr.shape == (4, 4)
        np.testing.assert_allclose(arr, want)


@pytest.mark.parametrize("leaf", ["dense", "diagonal", "identity", "symmetric", "causal"])
def test_3_10_C3_refined_matmul_stays_in_the_engine(leaf):
    """Review 3.10 C-3: each cut tile of a refined product was exported to NumPy and imported
    again (export ceiling, RAM), and cut structured leaves raised NotImplementedError."""
    from pycauset._internal import export_guard

    n = 4
    x = np.arange(1.0, 1.0 + n * n).reshape(n, n)
    leaves = {
        "dense": (lambda: pc.matrix(x), x),
        "diagonal": (lambda: pc.diagonal(np.array([1.0, 2.0, 3.0, 4.0])), np.diag([1.0, 2.0, 3.0, 4.0])),
        "identity": (lambda: pc.IdentityMatrix(n), np.eye(n)),
        "symmetric": (lambda: pc.symmetric(x + x.T), x + x.T),
        "causal": (lambda: pc.causal_matrix(np.triu(np.ones((n, n), bool), 1)), np.triu(np.ones((n, n)), 1)),
    }
    make, ref = leaves[leaf]
    G = pc.matrix([[make()]])
    X = pc.matrix(x)
    split = pc.matrix([[X[:2, :]], [X[2:, :]]])
    before = export_guard.export_count
    got = G @ split
    out = np.asarray(got)
    assert export_guard.export_count - before <= 1  # the final np.asarray only
    np.testing.assert_allclose(out, ref @ x)


def test_3_10_C4_block_results_use_the_grids_dtype():
    """Review 3.10 C-4: tiles promoted on their own, so an int8 tile wrapped (-56) where NumPy on
    the assembled matrices gives 200, and an int8 product overflowed where the grid is float32."""
    a8 = pc.matrix(np.full((2, 2), 100, np.int8))
    f32 = pc.matrix(np.ones((2, 2), np.float32))
    L = pc.matrix([[a8, f32]])
    assert L.dtype == "float32"
    np.testing.assert_array_equal(np.asarray(L + L), np.asarray(L) + np.asarray(L))
    G = pc.matrix([[f32, a8]]) @ pc.matrix([[f32], [a8]])
    assert G.dtype == "float32"
    np.testing.assert_allclose(np.asarray(G), np.full((2, 2), 20002.0))
    c = pc.matrix(np.ones((2, 2), np.complex128))
    Z = pc.matrix([[c, f32]])
    assert Z.dtype == "complex_float32"  # the engine's rule (narrowest float precision)
    assert np.asarray(Z + Z).dtype == np.complex64
    b = pc.matrix(np.ones((2, 2), bool))
    i = pc.matrix(np.full((2, 2), 3, np.int16))
    B = pc.matrix([[b, i]])
    np.testing.assert_array_equal(np.asarray(B * B), np.asarray(B) * np.asarray(B))


def test_3_10_C6_block_exports_go_through_the_one_guard():
    """allow_huge did not reach thunk tiles, thunk tiles were not counted against the ceiling,
    and copy=False returned a copy (NumPy 2's protocol raises)."""
    from pycauset._internal import export_guard
    from pycauset._internal.blockmatrix import BlockMatrix

    a = pc.matrix(np.ones((4, 4)))
    b = pc.matrix(np.full((4, 4), 2.0))
    product = BlockMatrix([[a, a]]) @ BlockMatrix([[b], [b]])  # one 4 x 4 thunk tile: 128 bytes
    grid = BlockMatrix([[a, b]])  # 256 bytes
    old = export_guard.get_max_bytes()
    try:
        export_guard.set_max_bytes(100)
        for obj in (product, grid, product.get_block(0, 0)):
            with pytest.raises(RuntimeError, match="ceiling"):
                np.asarray(obj)
            assert pc.to_numpy(obj, allow_huge=True).shape == obj.shape
        np.testing.assert_array_equal(pc.to_numpy(product, allow_huge=True), np.full((4, 4), 16.0))
        export_guard.set_max_bytes(200)
        with pytest.raises(RuntimeError, match="ceiling"):
            np.asarray(BlockMatrix([[product, product]]))  # two 128-byte thunk tiles
    finally:
        export_guard.set_max_bytes(old)
    for obj in (product, grid, product.get_block(0, 0)):
        with pytest.raises(ValueError, match="copy"):
            np.asarray(obj, copy=False)
    np.testing.assert_array_equal(np.asarray(grid, dtype=np.float32), np.hstack([np.ones((4, 4)), np.full((4, 4), 2.0)]))
