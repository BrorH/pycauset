import pytest

import pycauset
from pycauset._internal.blockmatrix import BlockMatrix
from pycauset._internal.submatrix_view import SubmatrixView


def test_refine_partitions_on_single_block_splits_into_views():
    A = pycauset.matrix(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    )
    M = BlockMatrix([[A]])

    R = M.refine_partitions(row_partitions=[0, 2, 4], col_partitions=[0, 1, 4])
    assert R.shape == (4, 4)
    assert R.block_rows == 2
    assert R.block_cols == 2

    # All refined blocks should be views (since we split A).
    assert isinstance(R.get_block(0, 0), SubmatrixView)
    assert isinstance(R.get_block(0, 1), SubmatrixView)
    assert isinstance(R.get_block(1, 0), SubmatrixView)
    assert isinstance(R.get_block(1, 1), SubmatrixView)

    for i in range(4):
        for j in range(4):
            assert R.get(i, j) == A.get(i, j)


def test_refine_partitions_on_2x2_blockmatrix_to_unit_tiles():
    # 2x2 blocks, each 2x2 => total 4x4
    A = pycauset.matrix([[1, 2], [3, 4]])
    B = pycauset.matrix([[5, 6], [7, 8]])
    C = pycauset.matrix([[9, 10], [11, 12]])
    D = pycauset.matrix([[13, 14], [15, 16]])
    M = BlockMatrix([[A, B], [C, D]])

    # Refine so each original 2x2 becomes 1x1 tiles. Must include original boundaries 0,2,4.
    R = M.refine_partitions(
        row_partitions=[0, 1, 2, 3, 4],
        col_partitions=[0, 1, 2, 3, 4],
    )

    assert R.block_rows == 4
    assert R.block_cols == 4

    # Every refined tile is contained within one original block; should be a view.
    assert isinstance(R.get_block(0, 0), SubmatrixView)
    assert isinstance(R.get_block(3, 3), SubmatrixView)

    for i in range(4):
        for j in range(4):
            assert R.get(i, j) == M.get(i, j)


def test_refine_partitions_requires_superset_of_existing_boundaries():
    A = pycauset.matrix([[1, 2], [3, 4]])
    B = pycauset.matrix([[5, 6], [7, 8]])
    M = BlockMatrix([[A, B]])  # row partitions [0,2], col partitions [0,2,4]

    # Missing existing boundary 2 in col partitions.
    with pytest.raises(ValueError, match=r"col_partitions must include all existing boundaries"):
        M.refine_partitions(row_partitions=[0, 2], col_partitions=[0, 1, 4])


def test_refine_partitions_validates_monotonic_and_endpoints():
    A = pycauset.matrix([[1, 2], [3, 4]])
    M = BlockMatrix([[A]])

    with pytest.raises(ValueError, match=r"row_partitions must start at 0 and end"):
        M.refine_partitions(row_partitions=[1, 2], col_partitions=[0, 2])

    with pytest.raises(ValueError, match=r"col_partitions must be strictly increasing"):
        M.refine_partitions(row_partitions=[0, 2], col_partitions=[0, 2, 2])
