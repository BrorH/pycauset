import pytest

import threading
import time

from pycauset._internal.blockmatrix import BlockMatrix, block_matmul
from pycauset._internal.thunks import StaleThunkError, ThunkBlock


class ToyMatrix:
    def __init__(self, data, *, counters=None):
        self._data = [list(row) for row in data]
        self._rows = len(self._data)
        self._cols = len(self._data[0]) if self._rows else 0
        self._counters = counters if counters is not None else {"matmul": 0, "add": 0}

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    @property
    def shape(self):
        return (self._rows, self._cols)

    def get(self, i, j):
        return self._data[i][j]

    def __matmul__(self, other):
        self._counters["matmul"] += 1
        assert self.cols() == other.rows()
        out = []
        for i in range(self.rows()):
            row = []
            for j in range(other.cols()):
                s = 0
                for k in range(self.cols()):
                    s += self.get(i, k) * other.get(k, j)
                row.append(s)
            out.append(row)
        return ToyMatrix(out, counters=self._counters)

    def __add__(self, other):
        self._counters["add"] += 1
        assert self.shape == other.shape
        out = []
        for i in range(self.rows()):
            out.append([self.get(i, j) + other.get(i, j) for j in range(self.cols())])
        return ToyMatrix(out, counters=self._counters)


class VersionedScalarMatrix:
    def __init__(self, value: int, *, version: int = 0):
        self._value = int(value)
        self._version = int(version)

    def rows(self):
        return 1

    def cols(self):
        return 1

    @property
    def shape(self):
        return (1, 1)

    @property
    def version(self) -> int:
        return int(self._version)

    def bump(self) -> None:
        self._version += 1

    def get(self, i, j):
        assert i == 0 and j == 0
        return self._value

    def __matmul__(self, other):
        return VersionedScalarMatrix(self.get(0, 0) * other.get(0, 0))

    def __add__(self, other):
        return VersionedScalarMatrix(self.get(0, 0) + other.get(0, 0))


def test_block_matmul_creates_thunks_and_is_lazy_on_repr():
    counters = {"matmul": 0, "add": 0}

    A0 = ToyMatrix([[1]], counters=counters)
    A1 = ToyMatrix([[2]], counters=counters)
    B0 = ToyMatrix([[3]], counters=counters)
    B1 = ToyMatrix([[4]], counters=counters)

    left = BlockMatrix([[A0, A1]])   # 1x2
    right = BlockMatrix([[B0], [B1]])  # 2x1

    out = block_matmul(left, right)

    blk = out.get_block(0, 0)
    assert isinstance(blk, ThunkBlock)

    # Printing must not evaluate.
    _ = repr(out)
    _ = str(out)
    _ = repr(blk)
    assert counters["matmul"] == 0
    assert counters["add"] == 0


def test_block_matmul_evaluates_on_element_access_and_deterministic_k_order():
    counters = {"matmul": 0, "add": 0}

    A0 = ToyMatrix([[1]], counters=counters)
    A1 = ToyMatrix([[2]], counters=counters)
    B0 = ToyMatrix([[3]], counters=counters)
    B1 = ToyMatrix([[4]], counters=counters)

    left = BlockMatrix([[A0, A1]])
    right = BlockMatrix([[B0], [B1]])

    out = block_matmul(left, right)

    # Access triggers evaluation of the single output block:
    # acc = (A0@B0) + (A1@B1)
    assert out.get(0, 0) == (1 * 3) + (2 * 4)
    assert counters["matmul"] == 2
    assert counters["add"] == 1


def test_stale_thunk_raises_after_input_version_changes():
    counters = {"matmul": 0, "add": 0}

    A0 = ToyMatrix([[1]], counters=counters)
    A1 = ToyMatrix([[2]], counters=counters)
    B0 = ToyMatrix([[3]], counters=counters)
    B1 = ToyMatrix([[4]], counters=counters)

    left = BlockMatrix([[A0, A1]])
    right = BlockMatrix([[B0], [B1]])

    out = block_matmul(left, right)

    # Mutate left block matrix structure -> version bump.
    left.set_block(0, 0, A0)

    with pytest.raises(StaleThunkError):
        out.get(0, 0)


def test_stale_thunk_raises_when_leaf_version_changes_even_if_parent_does_not(tmp_path):
    A = VersionedScalarMatrix(2)
    B = VersionedScalarMatrix(3)
    left = BlockMatrix([[A]])
    right = BlockMatrix([[B]])

    out = block_matmul(left, right)
    assert out.get(0, 0) == 6

    # Mutate leaf in-place without touching BlockMatrix.set_block().
    A.bump()
    with pytest.raises(StaleThunkError):
        out.get(0, 0)


def test_thunkblock_single_eval_under_concurrency():
    calls = {"n": 0}
    calls_lock = threading.Lock()

    def compute():
        with calls_lock:
            calls["n"] += 1
        time.sleep(0.05)
        return ToyMatrix([[123]])

    thunk = ThunkBlock(rows=1, cols=1, compute=compute)

    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i: int):
        barrier.wait()
        results[i] = thunk.materialize()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    assert all(r is results[0] for r in results)
