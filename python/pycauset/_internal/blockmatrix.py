from __future__ import annotations

from bisect import bisect_right
import os
from typing import Any, Iterable

from .submatrix_view import SubmatrixView
from .thunks import ThunkBlock


_PYCAUSET_MODULE: Any | None = None


def _get_pycauset() -> Any:
    global _PYCAUSET_MODULE
    if _PYCAUSET_MODULE is None:
        import pycauset as _pycauset  # type: ignore

        _PYCAUSET_MODULE = _pycauset
    return _PYCAUSET_MODULE


def _leaf_matmul(a: Any, b: Any) -> Any:
    """Matmul between leaf blocks via the public dispatch boundary.

    This ensures property-aware matmul conversions (e.g. diagonal/triangular)
    still apply when BlockMatrix decomposes a global op into leaf ops.
    """

    try:
        pycauset = _get_pycauset()
        native_matrix_base = getattr(pycauset, "MatrixBase", None)
        if native_matrix_base is not None and isinstance(a, native_matrix_base) and isinstance(
            b, native_matrix_base
        ):
            return pycauset.matmul(a, b)
    except Exception:
        pass
    return a @ b


def _try_io_prefetch(obj: Any) -> None:
    """Best-effort IO accelerator prefetch for a matrix-like object."""

    try:
        get_acc = getattr(obj, "get_accelerator", None)
        if not callable(get_acc):
            return
        acc = get_acc()
        if acc is None:
            return

        get_bf = getattr(obj, "get_backing_file", None)
        if callable(get_bf):
            path = get_bf()
        else:
            path = getattr(obj, "backing_file", None)
        if not path:
            return

        try:
            size = int(os.path.getsize(path))
        except OSError:
            return
        if size <= 0:
            return
        acc.prefetch(0, size)
    except Exception:
        return


def _try_io_discard(obj: Any) -> None:
    """Best-effort IO accelerator discard for a matrix-like object."""

    try:
        get_acc = getattr(obj, "get_accelerator", None)
        if not callable(get_acc):
            return
        acc = get_acc()
        if acc is None:
            return

        get_bf = getattr(obj, "get_backing_file", None)
        if callable(get_bf):
            path = get_bf()
        else:
            path = getattr(obj, "backing_file", None)
        if not path:
            return

        try:
            size = int(os.path.getsize(path))
        except OSError:
            return
        if size <= 0:
            return
        acc.discard(0, size)
    except Exception:
        return


def _safe_rows_cols(obj: Any) -> tuple[int, int] | None:
    try:
        return int(obj.rows()), int(obj.cols())
    except Exception:
        try:
            shape = getattr(obj, "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2:
                return int(shape[0]), int(shape[1])
        except Exception:
            pass
    return None


def _infer_dtype_label(obj: Any) -> str:
    dtype_attr = getattr(obj, "dtype", None)
    try:
        if dtype_attr is not None and not callable(dtype_attr):
            return str(dtype_attr)
    except Exception:
        pass

    name = type(obj).__name__
    # Common native type names.
    if "ComplexFloat16" in name:
        return "cf16"
    if "ComplexFloat32" in name:
        return "c32"
    if "ComplexFloat64" in name:
        return "c64"
    if "Float16" in name:
        return "f16"
    if "Float32" in name:
        return "f32"
    if name == "FloatMatrix" or name.endswith("FloatMatrix"):
        return "f64"
    if "Int8" in name:
        return "i8"
    if "Int16" in name:
        return "i16"
    if "Int64" in name:
        return "i64"
    if "UInt8" in name:
        return "u8"
    if "UInt16" in name:
        return "u16"
    if "UInt32" in name:
        return "u32"
    if "UInt64" in name:
        return "u64"
    if "Bit" in name:
        return "bit"
    return name


def _block_kind(obj: Any) -> str:
    if isinstance(obj, ThunkBlock):
        return "thunk"
    if isinstance(obj, BlockMatrix):
        return "block"
    return "leaf"


def _extend_staleness_sources(out: list[Any], blk: Any) -> None:
    """Add block + any underlying source objects for staleness pinning.

    Snapshot-at-creation semantics require stale detection even when callers
    mutate leaf blocks in-place without touching the parent BlockMatrix.
    """

    out.append(blk)
    if isinstance(blk, SubmatrixView):
        out.append(blk.source)


class BlockMatrix:
    """A structural matrix container (Phase B).

    This is an internal type used to represent a 2D grid of child matrices without
    densifying. Phase B guarantees construction validation, element indexing, and
    structure-only printing.

    It intentionally does NOT implement full compute semantics yet.
    """

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
        return "mixed"

    # --- integration helpers (Phase F) ---

    @staticmethod
    def _as_blockmatrix(obj: Any) -> "BlockMatrix":
        if isinstance(obj, BlockMatrix):
            return obj
        return BlockMatrix([[obj]])

    def __matmul__(self, other: Any) -> "BlockMatrix":
        return block_matmul(self, BlockMatrix._as_blockmatrix(other))

    def __rmatmul__(self, other: Any) -> "BlockMatrix":
        return block_matmul(BlockMatrix._as_blockmatrix(other), self)

    def __add__(self, other: Any) -> "BlockMatrix":
        return block_add(self, BlockMatrix._as_blockmatrix(other))

    def __radd__(self, other: Any) -> "BlockMatrix":
        return block_add(BlockMatrix._as_blockmatrix(other), self)

    def __sub__(self, other: Any) -> "BlockMatrix":
        return block_sub(self, BlockMatrix._as_blockmatrix(other))

    def __rsub__(self, other: Any) -> "BlockMatrix":
        return block_sub(BlockMatrix._as_blockmatrix(other), self)

    def __mul__(self, other: Any) -> "BlockMatrix":
        return block_mul(self, BlockMatrix._as_blockmatrix(other))

    def __rmul__(self, other: Any) -> "BlockMatrix":
        return block_mul(BlockMatrix._as_blockmatrix(other), self)

    def __truediv__(self, other: Any) -> "BlockMatrix":
        return block_div(self, BlockMatrix._as_blockmatrix(other))

    def __rtruediv__(self, other: Any) -> "BlockMatrix":
        return block_div(BlockMatrix._as_blockmatrix(other), self)

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

    # --- partition refinement (Phase C) ---

    def refine_partitions(
        self,
        *,
        row_partitions: list[int] | None = None,
        col_partitions: list[int] | None = None,
    ) -> "BlockMatrix":
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
        if not isinstance(target, list) or not all(isinstance(x, int) for x in target):
            raise TypeError(f"{axis}_partitions must be a list[int]")
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
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("indices must be integers")
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
        try:
            return blk[ii, jj]
        except Exception as e:
            raise TypeError("block does not support element indexing") from e

    def __getitem__(self, key: Any) -> Any:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col]")
        i, j = key
        return self.get(int(i), int(j))

    # --- printing ---

    def __repr__(self) -> str:
        return self._format(max_blocks=16)

    def __str__(self) -> str:
        return self._format(max_blocks=16)

    def _format(self, *, max_blocks: int) -> str:
        header = (
            f"BlockMatrix(shape={self.shape}, grid={self.block_rows}x{self.block_cols}, dtype=MIXED)"
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
            for c, blk in enumerate(row):
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

    # --- numpy interop (debug / fallback) ---

    def __array__(self, dtype: Any = None) -> Any:
        try:
            import numpy as np  # type: ignore

            out = np.empty(self.shape, dtype=dtype)
            r, c = self.shape
            for i in range(r):
                for j in range(c):
                    out[i, j] = self.get(i, j)
            return out
        except Exception as e:
            raise TypeError("BlockMatrix cannot be converted to a NumPy array") from e


def block_matmul(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """Internal block-matrix matmul orchestration (Phase D).

    Returns a BlockMatrix whose output blocks are lazy ThunkBlocks. Each output
    block sums over k in deterministic left-to-right order.
    """

    if left.cols() != right.rows():
        raise ValueError(
            f"matmul dimension mismatch: {left.rows()}x{left.cols()} @ {right.rows()}x{right.cols()}"
        )

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
                    prod = _leaf_matmul(a, b)
                    if acc is None:
                        acc = prod
                    else:
                        old_acc = acc
                        acc = acc + prod
                        _try_io_discard(old_acc)

                    _try_io_discard(prod)
                    _try_io_discard(a)
                    _try_io_discard(b)
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
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)


def block_add(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """Internal block-matrix elementwise addition orchestration (Phase D).

    Returns a BlockMatrix whose output blocks are lazy ThunkBlocks.
    Partitions are aligned by refinement to the union of row/col boundaries.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"add dimension mismatch: {left.rows()}x{left.cols()} + {right.rows()}x{right.cols()}"
        )

    rows_u = sorted(set(left.row_partitions) | set(right.row_partitions))
    cols_u = sorted(set(left.col_partitions) | set(right.col_partitions))

    l = left.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)
    r = right.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)

    out_blocks: list[list[Any]] = []
    for i in range(l.block_rows):
        out_row: list[Any] = []
        for j in range(l.block_cols):
            r0, r1 = l.row_partitions[i], l.row_partitions[i + 1]
            c0, c1 = l.col_partitions[j], l.col_partitions[j + 1]
            out_rows = r1 - r0
            out_cols = c1 - c0

            a0 = l.get_block(i, j)
            b0 = r.get_block(i, j)
            staleness_sources: list[Any] = [left, right]
            _extend_staleness_sources(staleness_sources, a0)
            _extend_staleness_sources(staleness_sources, b0)

            def _compute(a0: Any = a0, b0: Any = b0) -> Any:
                _try_io_prefetch(a0)
                _try_io_prefetch(b0)
                out = a0 + b0
                _try_io_discard(a0)
                _try_io_discard(b0)
                _try_io_discard(out)
                return out

            out_row.append(
                ThunkBlock(
                    rows=out_rows,
                    cols=out_cols,
                    compute=_compute,
                    sources_for_staleness=staleness_sources,
                    label="thunk",
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)


def block_sub(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """Internal block-matrix elementwise subtraction orchestration (Phase F).

    Returns a BlockMatrix whose output blocks are lazy ThunkBlocks.
    Partitions are aligned by refinement to the union of row/col boundaries.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"subtract dimension mismatch: {left.rows()}x{left.cols()} - {right.rows()}x{right.cols()}"
        )

    rows_u = sorted(set(left.row_partitions) | set(right.row_partitions))
    cols_u = sorted(set(left.col_partitions) | set(right.col_partitions))

    l = left.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)
    r = right.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)

    out_blocks: list[list[Any]] = []
    for i in range(l.block_rows):
        out_row: list[Any] = []
        for j in range(l.block_cols):
            r0, r1 = l.row_partitions[i], l.row_partitions[i + 1]
            c0, c1 = l.col_partitions[j], l.col_partitions[j + 1]
            out_rows = r1 - r0
            out_cols = c1 - c0

            a0 = l.get_block(i, j)
            b0 = r.get_block(i, j)
            staleness_sources: list[Any] = [left, right]
            _extend_staleness_sources(staleness_sources, a0)
            _extend_staleness_sources(staleness_sources, b0)

            def _compute(a0: Any = a0, b0: Any = b0) -> Any:
                _try_io_prefetch(a0)
                _try_io_prefetch(b0)
                out = a0 - b0
                _try_io_discard(a0)
                _try_io_discard(b0)
                _try_io_discard(out)
                return out

            out_row.append(
                ThunkBlock(
                    rows=out_rows,
                    cols=out_cols,
                    compute=_compute,
                    sources_for_staleness=staleness_sources,
                    label="thunk",
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)


def block_mul(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """Internal block-matrix elementwise multiply orchestration (Phase F)."""

    if left.shape != right.shape:
        raise ValueError(
            f"multiply dimension mismatch: {left.rows()}x{left.cols()} * {right.rows()}x{right.cols()}"
        )

    rows_u = sorted(set(left.row_partitions) | set(right.row_partitions))
    cols_u = sorted(set(left.col_partitions) | set(right.col_partitions))

    l = left.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)
    r = right.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)

    out_blocks: list[list[Any]] = []
    for i in range(l.block_rows):
        out_row: list[Any] = []
        for j in range(l.block_cols):
            r0, r1 = l.row_partitions[i], l.row_partitions[i + 1]
            c0, c1 = l.col_partitions[j], l.col_partitions[j + 1]
            out_rows = r1 - r0
            out_cols = c1 - c0

            a0 = l.get_block(i, j)
            b0 = r.get_block(i, j)
            staleness_sources: list[Any] = [left, right]
            _extend_staleness_sources(staleness_sources, a0)
            _extend_staleness_sources(staleness_sources, b0)

            def _compute(a0: Any = a0, b0: Any = b0) -> Any:
                _try_io_prefetch(a0)
                _try_io_prefetch(b0)
                out = a0 * b0
                _try_io_discard(a0)
                _try_io_discard(b0)
                _try_io_discard(out)
                return out

            out_row.append(
                ThunkBlock(
                    rows=out_rows,
                    cols=out_cols,
                    compute=_compute,
                    sources_for_staleness=staleness_sources,
                    label="thunk",
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)


def block_div(left: BlockMatrix, right: BlockMatrix) -> BlockMatrix:
    """Internal block-matrix elementwise divide orchestration (Phase F)."""

    if left.shape != right.shape:
        raise ValueError(
            f"divide dimension mismatch: {left.rows()}x{left.cols()} / {right.rows()}x{right.cols()}"
        )

    rows_u = sorted(set(left.row_partitions) | set(right.row_partitions))
    cols_u = sorted(set(left.col_partitions) | set(right.col_partitions))

    l = left.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)
    r = right.refine_partitions(row_partitions=rows_u, col_partitions=cols_u)

    out_blocks: list[list[Any]] = []
    for i in range(l.block_rows):
        out_row: list[Any] = []
        for j in range(l.block_cols):
            r0, r1 = l.row_partitions[i], l.row_partitions[i + 1]
            c0, c1 = l.col_partitions[j], l.col_partitions[j + 1]
            out_rows = r1 - r0
            out_cols = c1 - c0

            a0 = l.get_block(i, j)
            b0 = r.get_block(i, j)
            staleness_sources: list[Any] = [left, right]
            _extend_staleness_sources(staleness_sources, a0)
            _extend_staleness_sources(staleness_sources, b0)

            def _compute(a0: Any = a0, b0: Any = b0) -> Any:
                _try_io_prefetch(a0)
                _try_io_prefetch(b0)
                out = a0 / b0
                _try_io_discard(a0)
                _try_io_discard(b0)
                _try_io_discard(out)
                return out

            out_row.append(
                ThunkBlock(
                    rows=out_rows,
                    cols=out_cols,
                    compute=_compute,
                    sources_for_staleness=staleness_sources,
                    label="thunk",
                )
            )
        out_blocks.append(out_row)

    return BlockMatrix(out_blocks)
