import pytest

from pycauset._internal.blockmatrix import BlockMatrix, block_add
from pycauset._internal.thunks import StaleThunkError, ThunkBlock


class ToyMatrix:
    def __init__(self, data, *, counters=None):
        self._data = [list(row) for row in data]
        self._rows = len(self._data)
        self._cols = len(self._data[0]) if self._rows else 0
        self._counters = counters if counters is not None else {"add": 0}

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    @property
    def shape(self):
        return (self._rows, self._cols)

    def get(self, i, j):
        return self._data[i][j]

    def __add__(self, other):
        self._counters["add"] += 1
        assert self.shape == other.shape
        out = []
        for i in range(self.rows()):
            out.append([self.get(i, j) + other.get(i, j) for j in range(self.cols())])
        return ToyMatrix(out, counters=self._counters)


class VersionedToyMatrix(ToyMatrix):
    def __init__(self, data, *, counters=None, version: int = 0):
        super().__init__(data, counters=counters)
        self._version = int(version)

    @property
    def version(self) -> int:
        return int(self._version)

    def bump(self) -> None:
        self._version += 1


def test_block_add_creates_thunks_and_is_lazy_on_repr():
    counters = {"add": 0}

    A = ToyMatrix([[1, 2], [3, 4]], counters=counters)
    B = ToyMatrix([[10, 20], [30, 40]], counters=counters)

    left = BlockMatrix([[A]])
    right = BlockMatrix([[B]])

    out = block_add(left, right)
    blk = out.get_block(0, 0)
    assert isinstance(blk, ThunkBlock)

    _ = repr(out)
    _ = str(out)
    _ = repr(blk)
    assert counters["add"] == 0


def test_block_add_evaluates_on_element_access():
    counters = {"add": 0}

    A = ToyMatrix([[1, 2], [3, 4]], counters=counters)
    B = ToyMatrix([[10, 20], [30, 40]], counters=counters)

    out = block_add(BlockMatrix([[A]]), BlockMatrix([[B]]))

    assert out.get(0, 0) == 11
    assert out.get(1, 1) == 44
    assert counters["add"] == 1


def test_block_add_aligns_partitions_by_refinement_union():
    counters = {"add": 0}

    # Same overall 2x2, but left is refined into 1x1 tiles while right is 1 block.
    A = ToyMatrix([[1, 2], [3, 4]], counters=counters)
    B = ToyMatrix([[10, 20], [30, 40]], counters=counters)

    left = BlockMatrix([[A]]).refine_partitions(row_partitions=[0, 1, 2], col_partitions=[0, 1, 2])
    right = BlockMatrix([[B]])

    out = block_add(left, right)

    assert out.block_rows == 2
    assert out.block_cols == 2

    for i in range(2):
        for j in range(2):
            assert out.get(i, j) == (A.get(i, j) + B.get(i, j))


def test_block_add_stale_thunk_raises_after_input_version_changes():
    counters = {"add": 0}

    A = ToyMatrix([[1, 2], [3, 4]], counters=counters)
    B = ToyMatrix([[10, 20], [30, 40]], counters=counters)

    left = BlockMatrix([[A]])
    right = BlockMatrix([[B]])

    out = block_add(left, right)

    left.set_block(0, 0, A)

    with pytest.raises(StaleThunkError):
        out.get(0, 0)


def test_block_add_stale_thunk_raises_when_leaf_version_changes_even_if_parent_does_not():
    counters = {"add": 0}

    A = VersionedToyMatrix([[1, 2], [3, 4]], counters=counters)
    B = ToyMatrix([[10, 20], [30, 40]], counters=counters)

    left = BlockMatrix([[A]]).refine_partitions(row_partitions=[0, 1, 2], col_partitions=[0, 1, 2])
    right = BlockMatrix([[B]])

    out = block_add(left, right)
    assert out.get(0, 0) == 11

    # Mutate leaf in-place; no BlockMatrix.set_block() call.
    A.bump()
    with pytest.raises(StaleThunkError):
        out.get(0, 0)
