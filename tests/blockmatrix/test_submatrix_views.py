import pytest

import pycauset
from pycauset._internal.blockmatrix import BlockMatrix
from pycauset._internal.submatrix_view import SubmatrixView


class CountingMatrix:
    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self.get_calls = 0

    def rows(self) -> int:
        return self._rows

    def cols(self) -> int:
        return self._cols

    @property
    def shape(self):
        return (self._rows, self._cols)

    def get(self, i: int, j: int):
        self.get_calls += 1
        return (i, j)


def test_submatrix_view_dense_get_matches_source():
    A = pycauset.matrix(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ]
    )

    V = SubmatrixView(A, 1, 1, 2, 2)
    assert V.shape == (2, 2)
    assert V.get(0, 0) == A.get(1, 1)
    assert V.get(1, 1) == A.get(2, 2)
    assert V[0, 1] == A.get(1, 2)


def test_submatrix_view_blockmatrix_cross_block_get_matches_source():
    A = pycauset.matrix([[1, 2], [3, 4]])
    B = pycauset.matrix([[5], [6]])
    C = pycauset.matrix([[7, 8]])
    D = pycauset.matrix([[9]])

    M = BlockMatrix([[A, B], [C, D]])  # overall 3x3

    # A cross-block view: (row,col)=(1,1) hits A then B/C/D depending on element.
    V = SubmatrixView(M, 1, 1, 2, 2)
    assert V.shape == (2, 2)

    for i in range(2):
        for j in range(2):
            assert V.get(i, j) == M.get(1 + i, 1 + j)


def test_submatrix_view_composes_view_of_view():
    A = pycauset.matrix(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
    )

    V1 = SubmatrixView(A, 1, 1, 2, 3)
    V2 = SubmatrixView(V1, 0, 1, 2, 2)

    assert V2.shape == (2, 2)
    assert V2.source is A
    assert V2.row_offset == 1
    assert V2.col_offset == 2
    assert V2.get(1, 1) == A.get(2, 3)


def test_submatrix_view_repr_is_structure_only():
    C = CountingMatrix(10, 20)
    V = SubmatrixView(C, 2, 3, 4, 5)

    s = repr(V)
    assert "SubmatrixView" in s
    assert C.get_calls == 0


def test_submatrix_view_bounds_validation():
    A = pycauset.matrix([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        SubmatrixView(A, 0, 0, 3, 1)

    with pytest.raises(ValueError):
        SubmatrixView(A, 0, 0, 1, 3)

    with pytest.raises(ValueError):
        SubmatrixView(A, -1, 0, 1, 1)

    with pytest.raises(IndexError):
        SubmatrixView(A, 0, 0, 2, 2).get(2, 0)
