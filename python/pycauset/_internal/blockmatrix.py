from __future__ import annotations

import operator
from bisect import bisect_right
from typing import Any, Iterable

from .submatrix_view import SubmatrixView
from .thunks import ThunkBlock
from .coercion import safe_rows_cols as _safe_rows_cols

_PYCAUSET_MODULE: Any | None = None


def _get_pycauset() -> Any:
    global _PYCAUSET_MODULE
    if _PYCAUSET_MODULE is None:
        import pycauset as _pycauset  # type: ignore

        _PYCAUSET_MODULE = _pycauset
    return _PYCAUSET_MODULE


def _leaf_operand(x: Any) -> Any:
    """A leaf as an operand of the engine.

    - A thunk (the result of an earlier block operation) is evaluated: `(L @ R) @ R` raised
      TypeError, ThunkBlock has no operators (review 3.10 C-1).
    - A cut tile (`SubmatrixView`) of an engine leaf is the engine's slice: a zero-copy window of
      a dense leaf, a dense copy of just the tile of a structured one (3.7 C-13; the whole leaf
      was written out first). The tile went through NumPy (export ceiling, two RAM copies), and
      raised for structured leaves and for every element-wise op but `view + engine` (3.10
      C-2, C-3). A tile of a nested grid is its sub-grid (`BlockMatrix[r0:r1, c0:c1]`).
    - A tile of anything else (a duck-typed matrix-like) has no engine form: TypeError. (It was
      copied element by element into a Python list-of-lists matrix, review 3.10 C-12.)
    """
    if isinstance(x, ThunkBlock):
        return x.materialize()
    if isinstance(x, SubmatrixView):
        src = _leaf_operand(x.source)
        pc = _get_pycauset()
        r0, c0 = x.row_offset, x.col_offset
        if isinstance(src, (pc.MatrixBase, BlockMatrix)):
            return src[r0 : r0 + x.rows(), c0 : c0 + x.cols()]
        raise TypeError(f"a cut tile of a {type(src).__name__} leaf cannot be computed: block leaves that "
                        "operations cut must be PyCauset matrices (or block matrices)")
    return x


_DTYPES = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16",
           "float32", "float64", "complex_float16", "complex_float32", "complex_float64"}


def _np() -> Any:
    import numpy

    return numpy


def _is_scalar(x: Any) -> bool:
    np = _np()
    return (isinstance(x, (bool, int, float, complex, np.generic))
            or (isinstance(x, np.ndarray) and x.ndim == 0))


_SCALAR_OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}


def _promote(op: str, a: str | None, b: str | None) -> str | None:
    """The engine's result dtype of `a op b` (promotion::resolve); None when an operand is a
    matrix-like without a PyCauset dtype (then each tile keeps its own operation)."""
    if a not in _DTYPES or b not in _DTYPES:
        return None
    return _get_pycauset()._native._debug_resolve_promotion(op, a, b)["result_dtype"]


def _cast_leaf(x: Any, dtype: str | None) -> Any:
    """An engine leaf converted to the operation's result dtype, in the engine (cast_to). Every
    tile computes in the grid's result dtype, so a block result equals the same operation on the
    assembled matrices; tiles promoted on their own (an int8 tile wrapped at 200 where the grid is
    float32, review 3.10 C-4). Nested grids cast their own leaves."""
    pc = _get_pycauset()
    if dtype is None or not isinstance(x, pc.MatrixBase) or x.dtype == dtype:
        return x
    return pc._native._cast(x, dtype)


def _leaf_matmul(a: Any, b: Any, dtype: str | None = None) -> Any:
    """Matmul of two leaf blocks: engine operands through the public entry
    (`pycauset.matmul`), so the engine's structure rules apply to each leaf product. Chosen by
    type, not by trying one path and swallowing its error."""
    a, b = _cast_leaf(_leaf_operand(a), dtype), _cast_leaf(_leaf_operand(b), dtype)
    pc = _get_pycauset()
    engine = (pc.MatrixBase, SubmatrixView)
    if isinstance(a, engine) and isinstance(b, engine):
        return pc.matmul(a, b)
    return a @ b  # block operands recurse; other matrix-likes use their own operator


def _leaf_elementwise(a: Any, b: Any, op: str, dtype: str | None = None) -> Any:
    """Element-wise op of two leaf blocks: the operators (one engine path, F-086)."""
    a, b = _cast_leaf(_leaf_operand(a), dtype), _cast_leaf(_leaf_operand(b), dtype)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b
    raise ValueError(f"unknown elementwise op: {op}")


def _io_hint(obj: Any, action: str) -> None:
    """Prefetch / discard hint for a leaf's whole storage (native matrices only)."""
    hint = getattr(obj, f"_io_{action}", None)
    if callable(hint):
        hint()


def _try_io_prefetch(obj: Any) -> None:
    _io_hint(obj, "prefetch")


def _try_io_discard(obj: Any) -> None:
    _io_hint(obj, "discard")


def _leaf_dtype(obj: Any) -> str | None:
    if isinstance(obj, SubmatrixView):
        return _leaf_dtype(obj.source)
    d = getattr(obj, "dtype", None)
    return None if d is None else str(d)


def _infer_dtype_label(obj: Any) -> str:
    dtype_attr = getattr(obj, "dtype", None)
    if dtype_attr is not None and not callable(dtype_attr):
        return str(dtype_attr)
    return type(obj).__name__


def _block_kind(obj: Any) -> str:
    if isinstance(obj, ThunkBlock):
        return "thunk"
    if isinstance(obj, BlockMatrix):
        return "block"
    return "leaf"


def _extend_staleness_sources(out: list[Any], blk: Any) -> None:
    """Add a leaf and everything its values come from, for staleness pinning: the source of a
    cut tile, and every leaf of a nested grid (recursively). A write to any of them, before or
    after the first evaluation, makes the thunk stale. A nested grid's leaves were not pinned,
    so `[[[[a]]]] @ [[b]]` computed with a value written after creation where the flat grid
    raised (review 3.10 C-5). Only the leaves a tile uses are expanded, not whole operand grids."""
    out.append(blk)
    if isinstance(blk, SubmatrixView):
        _extend_staleness_sources(out, blk.source)
    elif isinstance(blk, BlockMatrix):
        for row in blk._blocks:
            for leaf in row:
                _extend_staleness_sources(out, leaf)


class BlockMatrix:
    """A matrix as a 2-D grid of child matrices (leaves), never densified as a whole.

    One dtype for the grid (the promotion over its leaves). Operations (`@`, `+ - * /` with
    a matrix or a scalar, `-`, `.T` / `.conj()` / `.H`) return block matrices of lazy
    `ThunkBlock`s; `bm[i, j]` reads an element and `bm[r0:r1, c0:c1]` is a sub-grid.
    Printing shows the structure only (internals/Block Matrices.md).
    """

    # NumPy defers to the block operators (`ndarray @ bm` exported the grid and computed in
    # NumPy, review 3.10 C-7), as for the engine classes.
    __array_ufunc__ = None
    __array_priority__ = 1e6

    def __init__(self, blocks: Iterable[Iterable[Any]]):
        grid = [list(row) for row in blocks]
        if not grid:
            raise ValueError("BlockMatrix blocks must be non-empty")
        if any(len(row) == 0 for row in grid):
            raise ValueError("BlockMatrix blocks must not contain empty rows")

        block_cols = len(grid[0])
        if any(len(row) != block_cols for row in grid):
            raise ValueError("BlockMatrix requires a rectangular block grid")

        # Validate shapes and compute row/col partitions.
        row_heights: list[int] = []
        col_widths: list[int] = [0 for _ in range(block_cols)]

        for r, row in enumerate(grid):
            expected_row_h: int | None = None
            for c, blk in enumerate(row):
                shape = _safe_rows_cols(blk)
                if shape is None:
                    raise TypeError(
                        "BlockMatrix blocks must be matrix-like (rows/cols or shape)"
                    )
                h, w = shape
                if expected_row_h is None:
                    expected_row_h = h
                elif h != expected_row_h:
                    raise ValueError(
                        f"BlockMatrix invalid grid: block-row {r} has inconsistent heights"
                    )

                if col_widths[c] == 0:
                    col_widths[c] = w
                elif w != col_widths[c]:
                    raise ValueError(
                        f"BlockMatrix invalid grid: block-col {c} has inconsistent widths"
                    )

            if expected_row_h is None:
                raise ValueError("BlockMatrix blocks must be non-empty")
            row_heights.append(expected_row_h)

        if any(w <= 0 for w in col_widths):
            raise ValueError("BlockMatrix block widths must be positive")

        self._blocks: list[list[Any]] = grid
        self._row_partitions: list[int] = [0]
        for h in row_heights:
            self._row_partitions.append(self._row_partitions[-1] + int(h))

        self._col_partitions: list[int] = [0]
        for w in col_widths:
            self._col_partitions.append(self._col_partitions[-1] + int(w))

        self._version: int = 0

    # --- minimal matrix protocol ---

    def rows(self) -> int:
        return self._row_partitions[-1]

    def cols(self) -> int:
        return self._col_partitions[-1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows(), self.cols())

    @property
    def dtype(self) -> str:
        """One dtype for the grid: the engine's promotion over its leaves (as for the operands of
        an addition). It was "mixed", and every tile promoted on its own (review 3.10 C-4)."""
        out: str | None = None
        for row in self._blocks:
            for blk in row:
                d = _leaf_dtype(blk)
                if d not in _DTYPES:
                    return "mixed"  # a matrix-like leaf without a PyCauset dtype
                out = d if out is None else (out if out == d else _promote("add", out, d))
        return out or "float64"

    # --- operators ---

    @staticmethod
    def _as_blockmatrix(obj: Any) -> BlockMatrix:
        if isinstance(obj, BlockMatrix):
            return obj
        pc = _get_pycauset()
        if not isinstance(obj, (pc.MatrixBase, pc.VectorBase, SubmatrixView, ThunkBlock)):
            obj = pc.matrix(obj)  # an ndarray or nested list: imported, not a NumPy leaf
        return BlockMatrix([[obj]])

    def _operand(self, other: Any) -> BlockMatrix:
        pc = _get_pycauset()
        if isinstance(other, pc.VectorBase) or (isinstance(other, _np().ndarray) and other.ndim == 1):
            raise TypeError("BlockMatrix does not take vector operands; use an n x 1 matrix")
        return BlockMatrix._as_blockmatrix(other)

    def _binary(self, other: Any, op: str, reflected: bool) -> BlockMatrix:
        if _is_scalar(other):  # mapped over the leaves, in the grid's dtype (review 3.10 C-8)
            f = _SCALAR_OPS[op]
            fn = (lambda x: f(other, x)) if reflected else (lambda x: f(x, other))
            return _block_map(self, fn, _scalar_dtype(self.dtype, fn))
        o = self._operand(other)
        return _block_elementwise(op, o, self) if reflected else _block_elementwise(op, self, o)

    def __matmul__(self, other: Any) -> BlockMatrix:
        return block_matmul(self, self._operand(other))

    def __rmatmul__(self, other: Any) -> BlockMatrix:
        return block_matmul(self._operand(other), self)

    def __add__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "add", False)

    def __radd__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "add", True)

    def __sub__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "sub", False)

    def __rsub__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "sub", True)

    def __mul__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "mul", False)

    def __rmul__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "mul", True)

    def __truediv__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "div", False)

    def __rtruediv__(self, other: Any) -> BlockMatrix:
        return self._binary(other, "div", True)

    def __neg__(self) -> BlockMatrix:
        fn = lambda x: -x  # noqa: E731
        return _block_map(self, fn, _scalar_dtype(self.dtype, fn))

    # --- transposes: the grid transposed, each leaf an engine view (review 3.10 C-8) ---

    @property
    def T(self) -> BlockMatrix:
        return _block_map(self, lambda x: x.T, self.dtype if self.dtype in _DTYPES else None, transpose=True)

    def transpose(self) -> BlockMatrix:
        return self.T

    def conj(self) -> BlockMatrix:
        return _block_map(self, lambda x: x.conj(), self.dtype if self.dtype in _DTYPES else None)

    @property
    def H(self) -> BlockMatrix:
        return _block_map(self, lambda x: x.conj().T, self.dtype if self.dtype in _DTYPES else None, transpose=True)

    # --- block introspection ---

    @property
    def block_rows(self) -> int:
        return len(self._blocks)

    @property
    def block_cols(self) -> int:
        return len(self._blocks[0])

    @property
    def row_partitions(self) -> list[int]:
        return list(self._row_partitions)

    @property
    def col_partitions(self) -> list[int]:
        return list(self._col_partitions)

    @property
    def version(self) -> int:
        return int(self._version)

    def get_block(self, r: int, c: int) -> Any:
        return self._blocks[r][c]

    def set_block(self, r: int, c: int, block: Any) -> None:
        shape = _safe_rows_cols(block)
        if shape is None:
            raise TypeError("set_block expects a matrix-like block")

        # Validate against existing block-row/col sizes.
        old = self._blocks[r][c]
        old_shape = _safe_rows_cols(old)
        if old_shape is None:
            raise RuntimeError("internal error: existing block has no shape")

        if shape != old_shape:
            raise ValueError(
                f"set_block shape mismatch: expected {old_shape[0]}x{old_shape[1]}, got {shape[0]}x{shape[1]}"
            )

        self._blocks[r][c] = block
        self._version += 1

    # --- partition refinement ---

    def refine_partitions(
        self,
        *,
        row_partitions: list[int] | None = None,
        col_partitions: list[int] | None = None,
    ) -> BlockMatrix:
        """Return a refined BlockMatrix whose grid is defined by partitions.

        The target partitions must be a refinement (superset) of the current
        `row_partitions`/`col_partitions`. This guarantees each refined block
        lies entirely within a single original block (no cross-block tiles).
        """

        target_rows = self._validate_refinement(
            target=row_partitions, current=self._row_partitions, total=self.rows(), axis="row"
        )
        target_cols = self._validate_refinement(
            target=col_partitions, current=self._col_partitions, total=self.cols(), axis="col"
        )

        refined_blocks: list[list[Any]] = []
        for r0, r1 in zip(target_rows[:-1], target_rows[1:]):
            row: list[Any] = []
            br = bisect_right(self._row_partitions, r0) - 1
            br0 = self._row_partitions[br]
            for c0, c1 in zip(target_cols[:-1], target_cols[1:]):
                bc = bisect_right(self._col_partitions, c0) - 1
                bc0 = self._col_partitions[bc]

                blk = self._blocks[br][bc]
                local_r0 = r0 - br0
                local_c0 = c0 - bc0
                sub_rows = r1 - r0
                sub_cols = c1 - c0

                shape = _safe_rows_cols(blk)
                if shape is None:
                    raise TypeError("BlockMatrix contains a non-matrix-like block")
                blk_rows, blk_cols = shape

                if local_r0 == 0 and local_c0 == 0 and sub_rows == blk_rows and sub_cols == blk_cols:
                    row.append(blk)
                else:
                    row.append(SubmatrixView(blk, local_r0, local_c0, sub_rows, sub_cols))
            refined_blocks.append(row)

        return BlockMatrix(refined_blocks)

    @staticmethod
    def _validate_refinement(
        *,
        target: list[int] | None,
        current: list[int],
        total: int,
        axis: str,
    ) -> list[int]:
        if target is None:
            return list(current)
        if not isinstance(target, (list, tuple)) or any(isinstance(x, bool) for x in target):
            raise TypeError(f"{axis}_partitions must be a list of integers")
        try:
            target = [operator.index(x) for x in target]  # NumPy integers too (review 3.10 C-8)
        except TypeError:
            raise TypeError(f"{axis}_partitions must be a list of integers") from None
        if len(target) < 2:
            raise ValueError(f"{axis}_partitions must have at least [0, {total}]")
        if target[0] != 0 or target[-1] != total:
            raise ValueError(f"{axis}_partitions must start at 0 and end at {total}")
        if any(b <= a for a, b in zip(target, target[1:])):
            raise ValueError(f"{axis}_partitions must be strictly increasing")

        target_set = set(target)
        missing = [x for x in current if x not in target_set]
        if missing:
            raise ValueError(
                f"{axis}_partitions must include all existing boundaries; missing {missing}"
            )
        return list(target)

    # --- element indexing ---

    def _find_block_index(self, i: int, j: int) -> tuple[int, int, int, int]:
        # NumPy's integers (operator.index: np.int64 works, a float does not) and negative
        # indices from the end (review 3.10 C-8).
        if isinstance(i, bool) or isinstance(j, bool):
            raise IndexError("boolean scalar indices are not supported")
        i, j = operator.index(i), operator.index(j)
        if i < 0:
            i += self.rows()
        if j < 0:
            j += self.cols()
        if i < 0 or j < 0 or i >= self.rows() or j >= self.cols():
            raise IndexError("index out of range")

        br = bisect_right(self._row_partitions, i) - 1
        bc = bisect_right(self._col_partitions, j) - 1
        i0 = self._row_partitions[br]
        j0 = self._col_partitions[bc]
        return br, bc, i - i0, j - j0

    def get(self, i: int, j: int) -> Any:
        br, bc, ii, jj = self._find_block_index(i, j)
        blk = self._blocks[br][bc]
        get_fn = getattr(blk, "get", None)
        if callable(get_fn):
            return get_fn(ii, jj)
        return blk[ii, jj]

    def __getitem__(self, key: Any) -> Any:
        """`bm[i, j]` an element; `bm[r0:r1, c0:c1]` (step 1) the sub-grid, its leaves the tiles
        of a refinement at the slice bounds (no copy; review 3.10 C-8). Other keys raise."""
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("BlockMatrix indices are [row, col] or [r0:r1, c0:c1]")
        i, j = key
        if isinstance(i, slice) and isinstance(j, slice):
            return self._subgrid(i, j)
        if isinstance(i, slice) or isinstance(j, slice):
            raise TypeError("BlockMatrix slicing takes a slice on both axes (bm[r0:r1, c0:c1])")
        return self.get(i, j)

    def _subgrid(self, rs: slice, cs: slice) -> BlockMatrix:
        r0, r1, rstep = rs.indices(self.rows())
        c0, c1, cstep = cs.indices(self.cols())
        if rstep != 1 or cstep != 1:
            raise TypeError("BlockMatrix slicing takes steps of 1")
        if r1 <= r0 or c1 <= c0:
            raise ValueError("BlockMatrix slice is empty")
        rows = sorted(set(self._row_partitions) | {r0, r1})
        cols = sorted(set(self._col_partitions) | {c0, c1})
        g = self.refine_partitions(row_partitions=rows, col_partitions=cols)
        bi = range(rows.index(r0), rows.index(r1))
        bj = range(cols.index(c0), cols.index(c1))
        return BlockMatrix([[g.get_block(i, j) for j in bj] for i in bi])

    # --- printing ---

    def __repr__(self) -> str:
        return self._format(max_blocks=16)

    def __str__(self) -> str:
        return self._format(max_blocks=16)

    def _format(self, *, max_blocks: int) -> str:
        header = (
            f"BlockMatrix(shape={self.shape}, grid={self.block_rows}x{self.block_cols}, dtype={self.dtype})"
        )
        parts = [
            header,
            f"row_partitions={self._row_partitions}",
            f"col_partitions={self._col_partitions}",
        ]

        # Simple grid preview (structure-only; no evaluation).
        shown = 0
        for r, row in enumerate(self._blocks):
            row_cells: list[str] = []
            for blk in row:
                if shown >= max_blocks:
                    row_cells.append("…")
                    continue
                shape = _safe_rows_cols(blk)
                shp = "?x?" if shape is None else f"{shape[0]}x{shape[1]}"
                row_cells.append(f"{shp}:{_infer_dtype_label(blk)}:{_block_kind(blk)}")
                shown += 1
            parts.append(f"[{r}] " + " | ".join(row_cells))
            if shown >= max_blocks:
                break

        return "\n".join(parts)

    # --- numpy interop (explicit conversion) ---

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """The assembled matrix, under the export guard (`export_to_numpy`)."""
        from . import export_guard

        return export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=copy is not False)

    def _export_numpy(self) -> Any:
        """The assembled matrix, block by block (the guard checked the whole grid)."""
        import numpy as np

        from . import export_guard

        arr = np.block([[export_guard.export_part(b) for b in row] for row in self._blocks])
        if self.dtype in _DTYPES:
            arr = arr.astype(self._numpy_dtype(), copy=False)  # the grid's (engine) dtype
        return arr

    def _numpy_dtype(self) -> str:
        """The grid's dtype as NumPy's (np.block promotes by NumPy's rules; the grid's dtype is the
        engine's, which keeps float32 with float64, review 3.10 C-4)."""
        from .matrix_api import _NUMPY_DTYPE

        return _NUMPY_DTYPE.get(self.dtype, "complex64")


def block_matmul(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """`left @ right` for two block matrices.

    Returns a BlockMatrix whose output blocks are lazy ThunkBlocks. Each output
    block sums over k in deterministic left-to-right order.
    """

    if left.cols() != right.rows():
        raise ValueError(
            f"matmul dimension mismatch: {left.rows()}x{left.cols()} @ {right.rows()}x{right.cols()}"
        )

    rdtype = _promote("matmul", left.dtype, right.dtype)
    inner = sorted(set(left.col_partitions) | set(right.row_partitions))
    left_r = left.refine_partitions(col_partitions=inner)
    right_r = right.refine_partitions(row_partitions=inner)

    # Output partitions are inherited (unrefined) along outer axes.
    out_blocks: list[list[Any]] = []
    for i in range(left_r.block_rows):
        out_row: list[Any] = []
        for j in range(right_r.block_cols):
            # Determine output block shape from partitions.
            r0, r1 = left_r.row_partitions[i], left_r.row_partitions[i + 1]
            c0, c1 = right_r.col_partitions[j], right_r.col_partitions[j + 1]
            out_rows = r1 - r0
            out_cols = c1 - c0

            pairs: list[tuple[Any, Any]] = []
            staleness_sources: list[Any] = [left, right]
            for k in range(left_r.block_cols):
                a = left_r.get_block(i, k)
                b = right_r.get_block(k, j)
                pairs.append((a, b))
                _extend_staleness_sources(staleness_sources, a)
                _extend_staleness_sources(staleness_sources, b)

            def _compute(pairs: list[tuple[Any, Any]] = pairs) -> Any:
                acc: Any | None = None
                for a, b in pairs:
                    _try_io_prefetch(a)
                    _try_io_prefetch(b)
                    prod = _leaf_matmul(a, b, rdtype)
                    if acc is None:
                        acc = prod
                    else:
                        old_acc = acc
                        acc = acc + prod
                        _try_io_discard(old_acc)

                    if prod is not acc:  # on the first term the accumulator is the product
                        _try_io_discard(prod)
                    # The inputs are not discarded: every other output block of this row /
                    # column reads them again (review 3.10 C-11).
                if acc is None:
                    raise RuntimeError("internal error: empty inner dimension")
                return acc

            out_row.append(
                ThunkBlock(
                    rows=out_rows,
                    cols=out_cols,
                    compute=_compute,
                    sources_for_staleness=staleness_sources,
                    label="thunk",
                    dtype=rdtype,
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)


def _block_elementwise(op: str, left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """`left op right` element-wise ("add", "sub", "mul", "div"): a BlockMatrix of
    lazy ThunkBlocks over the union of both grids' boundaries (each tile a leaf pair). One
    orchestrator for the four operations (they were four copies, review 3.10 C-12)."""
    if left.shape != right.shape:
        raise ValueError(f"{op} dimension mismatch: {left.rows()}x{left.cols()} and {right.rows()}x{right.cols()}")
    rdtype = _promote(op, left.dtype, right.dtype)
    rows_u = sorted(set(left.row_partitions) | set(right.row_partitions))
    cols_u = sorted(set(left.col_partitions) | set(right.col_partitions))
    lg = left.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)
    rg = right.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)

    out_blocks: list[list[Any]] = []
    for i in range(lg.block_rows):
        out_row: list[Any] = []
        for j in range(lg.block_cols):
            a0, b0 = lg.get_block(i, j), rg.get_block(i, j)
            sources: list[Any] = [left, right]
            _extend_staleness_sources(sources, a0)
            _extend_staleness_sources(sources, b0)

            def _compute(a0: Any = a0, b0: Any = b0) -> Any:
                _try_io_prefetch(a0)
                _try_io_prefetch(b0)
                return _leaf_elementwise(a0, b0, op, rdtype)  # inputs and the result stay resident (C-11)

            out_row.append(ThunkBlock(rows=rows_u[i + 1] - rows_u[i], cols=cols_u[j + 1] - cols_u[j], compute=_compute,
                                      sources_for_staleness=sources, label="thunk", dtype=rdtype))
        out_blocks.append(out_row)
    return BlockMatrix(out_blocks)


def _block_map(grid: BlockMatrix, fn: Any, dtype: str | None, *, transpose: bool = False) -> BlockMatrix:
    """A BlockMatrix of lazy ThunkBlocks `fn(leaf)` over every leaf (as an engine operand of the
    grid's dtype), the grid transposed when `transpose`. Scalars, negation and transposes of a
    grid (review 3.10 C-8: they raised)."""
    out_blocks: list[list[Any]] = []
    rows = [[grid.get_block(i, j) for j in range(grid.block_cols)] for i in range(grid.block_rows)]
    if transpose:
        rows = [list(col) for col in zip(*rows)]
    gdtype = grid.dtype if grid.dtype in _DTYPES else None
    for row in rows:
        out_row: list[Any] = []
        for blk in row:
            r, c = _safe_rows_cols(blk)
            sources: list[Any] = [grid]
            _extend_staleness_sources(sources, blk)

            def _compute(blk: Any = blk) -> Any:
                return fn(_leaf_operand(_cast_leaf(blk, gdtype)))

            out_row.append(ThunkBlock(rows=c if transpose else r, cols=r if transpose else c, compute=_compute,
                                      sources_for_staleness=sources, label="thunk", dtype=dtype))
        out_blocks.append(out_row)
    return BlockMatrix(out_blocks)


def _scalar_dtype(grid_dtype: str | None, fn: Any) -> str | None:
    """The dtype `fn` gives for a leaf of the grid's dtype (the engine decides, on a 1 x 1 probe)."""
    if grid_dtype not in _DTYPES:
        return None
    pc = _get_pycauset()
    return str(fn(pc.zeros((1, 1), dtype=grid_dtype)).dtype)


def block_add(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    return _block_elementwise("add", left, right)


def block_sub(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    return _block_elementwise("sub", left, right)


def block_mul(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    return _block_elementwise("mul", left, right)


def block_div(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    return _block_elementwise("div", left, right)
