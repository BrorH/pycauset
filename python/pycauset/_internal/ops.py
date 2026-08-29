from __future__ import annotations

import os
from typing import Any

from . import big_blob_cache as _big_blob_cache
from . import export_guard, io_observability
from . import linalg_cache as _linalg_cache
from . import properties as _props


def _track_and_mark_temporary_if_native(obj: Any, *, deps: OpsDeps) -> None:
    native_matrix_base = getattr(deps.native, "MatrixBase", None)
    native_vector_base = getattr(deps.native, "VectorBase", None)
    if native_matrix_base is not None and isinstance(obj, native_matrix_base):
        deps.track_matrix(obj)
        deps.mark_temporary_if_auto(obj)
    elif native_vector_base is not None and isinstance(obj, native_vector_base):
        deps.track_matrix(obj)
        deps.mark_temporary_if_auto(obj)


def _as_pycauset_array(obj: Any, *, deps: OpsDeps) -> Any:
    asarray = getattr(deps.native, "asarray", None)
    if asarray is None:
        raise RuntimeError("native.asarray is not available")
    out = asarray(obj)
    _track_and_mark_temporary_if_native(out, deps=deps)
    return out


def _as_pycauset_vector(obj: Any, *, deps: OpsDeps) -> Any:
    """Convert a 1-D NumPy array to a native vector, handling complex dtypes.

    native.asarray only supports real dtypes for 1-D arrays; complex 1-D
    (e.g. general eigenvalues) must go through the ComplexFloat64Vector ctor.
    """
    np_module = deps.np_module
    if np_module is not None and np_module.iscomplexobj(obj):
        cls = getattr(deps.native, "ComplexFloat64Vector", None)
        if cls is not None:
            out = cls(obj)
            _track_and_mark_temporary_if_native(out, deps=deps)
            return out
    return _as_pycauset_array(obj, deps=deps)


def _to_numpy_matrix(obj: Any, *, deps: OpsDeps, allow_huge: bool = False) -> Any:
    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for this operation")
    export_guard.ensure_export_allowed(
        obj,
        allow_huge=allow_huge,
        ceiling_bytes=export_guard.get_max_bytes(),
    )
    return np_module.asarray(obj)


class OpsDeps:
    def __init__(
        self,
        *,
        native: Any,
        np_module: Any | None,
        Matrix: Any,
        TriangularBitMatrix: Any,
        track_matrix: Any,
        mark_temporary_if_auto: Any,
        warnings_module: Any,
        io_observer: Any | None,
        streaming_manager: Any | None = None,
    ) -> None:
        self.native = native
        self.np_module = np_module
        self.Matrix = Matrix
        self.TriangularBitMatrix = TriangularBitMatrix
        self.track_matrix = track_matrix
        self.mark_temporary_if_auto = mark_temporary_if_auto
        self.warnings = warnings_module
        self.io_observer = io_observer
        self.streaming_manager = streaming_manager


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


def _effective_structure_for(obj: Any) -> str:
    # Native structural types report their structure by type, even when the
    # properties mapping is not populated. (Triangular types are intentionally not
    # mapped here: the matmul triangular fast path converts to float64, which is
    # wrong for bit/integer triangular matrices.)
    name = type(obj).__name__
    if name == "IdentityMatrix":
        return "identity"
    if name == "DiagonalMatrix":
        return "diagonal"
    # SymmetricMatrix / AntiSymmetricMatrix are native float64 structural types;
    # their packed upper-triangle storage enforces the structure by construction,
    # so the type name is authoritative here.
    if name == "SymmetricMatrix":
        return "symmetric"
    if name == "AntiSymmetricMatrix":
        return "antisymmetric"
    if name == "LazyAllocated":
        kind = getattr(obj, "kind", None)
        if kind == "zeros":
            return "zero"
        if kind == "ones":
            return "constant"
        return "general"
    try:
        props = _props.get_properties(obj)
        return _props.effective_structure_from_properties(props)
    except Exception:
        return "general"


def _record_io_trace(op_name: str, operands: list[Any], *, deps: OpsDeps, allow_huge: bool = False, supports_streaming: bool = True) -> None:
    manager = getattr(deps, "streaming_manager", None)
    record = None
    if manager is not None:
        try:
            record = manager.plan(op_name, operands, allow_huge=allow_huge)
        except Exception:
            record = None

    if record is None:
        observer = getattr(deps, "io_observer", None)
        if observer is None:
            return None
        try:
            record = observer.plan_and_record(op_name, operands, allow_huge=allow_huge)
        except Exception:
            return None

    # Ops that cannot stream (e.g. LAPACK eigen/factorizations) are always "direct".
    if not supports_streaming and isinstance(record, dict):
        record["route"] = "direct"
        record["reason"] = "op does not support streaming"
    return record


def _prefetch_if_streaming(record: Any, operands: list[Any], *, deps: OpsDeps | None = None) -> None:
    manager = getattr(deps, "streaming_manager", None) if deps is not None else None
    if manager is not None:
        try:
            manager.prefetch(record, operands)
            return
        except Exception:
            pass

    try:
        if record is None or record.get("route") != "streaming":
            return
    except Exception:
        return

    for obj in operands:
        io_observability._try_io_prefetch(obj)
    _append_event(record, event_type="io", detail="prefetch")


def _discard_if_streaming(
    record: Any, operands: list[Any], result: Any | None = None, *, deps: OpsDeps | None = None
) -> None:
    manager = getattr(deps, "streaming_manager", None) if deps is not None else None
    if manager is not None:
        try:
            manager.discard(record, operands, result)
            return
        except Exception:
            pass

    try:
        if record is None or record.get("route") != "streaming":
            return
    except Exception:
        return
    payloads = list(operands)
    if result is not None:
        payloads.append(result)
    io_observability._discard_if_streaming(record, payloads)
    _append_event(record, event_type="io", detail="discard")


def _annotate_impl(record: Any, label: str, *, deps: OpsDeps | None = None) -> None:
    manager = getattr(deps, "streaming_manager", None) if deps is not None else None
    if manager is not None:
        try:
            manager.annotate_impl(record, label)
            return
        except Exception:
            pass

    try:
        if isinstance(record, dict):
            record["impl"] = label
            _append_event(record, event_type="compute", detail=f"impl={label}")
    except Exception:
        return


def _append_event(record: Any, *, event_type: str, detail: str) -> None:
    try:
        if not isinstance(record, dict):
            return
        events = record.setdefault("events", [])
        events.append({"type": event_type, "detail": detail})
    except Exception:
        return


def _streaming_invert(matrix: Any, *, deps: OpsDeps, rec: Any) -> Any | None:
    np_module = deps.np_module
    if np_module is None:
        return None

    shape = _safe_rows_cols(matrix)
    if shape is None:
        return None

    try:
        data = np_module.array([[matrix.get(i, j) for j in range(shape[1])] for i in range(shape[0])])
    except Exception:
        return None

    try:
        inv_np = np_module.linalg.inv(data)
    except Exception:
        return None

    out = _as_pycauset_array(inv_np, deps=deps)
    _annotate_impl(rec, "streaming_python", deps=deps)
    return out


def _streaming_eigvalsh(matrix: Any, *, deps: OpsDeps, rec: Any) -> Any | None:
    np_module = deps.np_module
    if np_module is None:
        return None

    shape = _safe_rows_cols(matrix)
    if shape is None:
        return None

    try:
        data = np_module.array([[matrix.get(i, j) for j in range(shape[1])] for i in range(shape[0])])
    except Exception:
        return None

    try:
        vals = np_module.linalg.eigvalsh(data)
    except Exception:
        return None

    out = _as_pycauset_array(vals, deps=deps)
    _annotate_impl(rec, "streaming_python", deps=deps)
    return out


def _streaming_eigh(matrix: Any, *, deps: OpsDeps, rec: Any) -> tuple[Any, Any] | None:
    np_module = deps.np_module
    if np_module is None:
        return None

    shape = _safe_rows_cols(matrix)
    if shape is None:
        return None

    try:
        data = np_module.array([[matrix.get(i, j) for j in range(shape[1])] for i in range(shape[0])])
    except Exception:
        return None

    try:
        w, v = np_module.linalg.eigh(data)
    except Exception:
        return None

    w_out = _as_pycauset_array(w, deps=deps)
    v_out = _as_pycauset_array(v, deps=deps)
    _annotate_impl(rec, "streaming_python", deps=deps)
    return w_out, v_out


def _streaming_eigvals_arnoldi(matrix: Any, k: int, m: int, tol: float, *, deps: OpsDeps, rec: Any) -> Any | None:
    np_module = deps.np_module
    if np_module is None:
        return None

    shape = _safe_rows_cols(matrix)
    if shape is None:
        return None

    try:
        data = np_module.array([[matrix.get(i, j) for j in range(shape[1])] for i in range(shape[0])])
    except Exception:
        return None

    try:
        eigs = np_module.linalg.eigvals(data)
        if np_module.iscomplexobj(eigs):
            if not np_module.allclose(eigs.imag, 0.0, atol=tol):
                return None
            eigs = eigs.real
        eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
        top = np_module.array(eigs_sorted[:k])
    except Exception:
        return None

    out = _as_pycauset_array(top, deps=deps)
    _annotate_impl(rec, "streaming_python", deps=deps)
    return out


def _streaming_matmul_tiles(a: Any, b: Any, *, deps: OpsDeps, rec: Any) -> Any | None:
    np_module = deps.np_module
    if np_module is None:
        return None

    shape_a = _safe_rows_cols(a)
    shape_b = _safe_rows_cols(b)
    if shape_a is None or shape_b is None:
        return None
    a_rows, a_cols = shape_a
    b_rows, b_cols = shape_b
    if a_cols != b_rows:
        return None

    tile = rec.get("tile_shape") if isinstance(rec, dict) else None
    try:
        t_r, t_c = int(tile[0]), int(tile[1]) if tile is not None else (64, 64)
    except Exception:
        t_r, t_c = 64, 64

    res = deps.Matrix(np_module.zeros((a_rows, b_cols), dtype=float))

    set_fn = getattr(res, "set", None)

    for i0 in range(0, a_rows, t_r):
        i1 = min(i0 + t_r, a_rows)
        for j0 in range(0, b_cols, t_c):
            j1 = min(j0 + t_c, b_cols)
            block = np_module.zeros((i1 - i0, j1 - j0), dtype=float)
            for k0 in range(0, a_cols, t_c):
                k1 = min(k0 + t_c, a_cols)

                a_tile = np_module.array(
                    [[a.get(i, k) for k in range(k0, k1)] for i in range(i0, i1)]
                )
                b_tile = np_module.array(
                    [[b.get(k, j) for j in range(j0, j1)] for k in range(k0, k1)]
                )
                block += np_module.matmul(a_tile, b_tile)

            for ii in range(i0, i1):
                for jj in range(j0, j1):
                    val = block[ii - i0, jj - j0]
                    if callable(set_fn):
                        set_fn(ii, jj, float(val))
                    else:
                        res[ii, jj] = float(val)

    _annotate_impl(rec, "streaming_python", deps=deps)
    return res


def _try_convert_to_diagonal_f64(obj: Any, *, deps: OpsDeps) -> Any | None:
    diag_cls = getattr(deps.native, "DiagonalMatrix", None)
    if diag_cls is None:
        return None

    shape = _safe_rows_cols(obj)
    if shape is None:
        return None
    rows, cols = shape
    if rows != cols:
        return None

    try:
        out = diag_cls(rows)
        for i in range(rows):
            out.set_diagonal(i, float(obj.get(i, i)))
        _track_and_mark_temporary_if_native(out, deps=deps)
        return out
    except Exception:
        return None


def _try_convert_to_triangular_f64(obj: Any, *, which: str, deps: OpsDeps) -> Any | None:
    tri_cls = getattr(deps.native, "TriangularFloatMatrix", None)
    if tri_cls is None:
        return None

    shape = _safe_rows_cols(obj)
    if shape is None:
        return None
    rows, cols = shape
    if rows != cols:
        return None

    lower = which == "lower_triangular"
    upper = which == "upper_triangular"
    if not (lower or upper):
        return None

    try:
        out = tri_cls(rows, True)
        if lower:
            try:
                out.set_transposed(True)
            except Exception:
                return None

        # Gospel semantics: treat out-of-triangle entries as zero.
        for i in range(rows):
            if upper:
                j0, j1 = i, cols
            else:
                j0, j1 = 0, i + 1
            for j in range(j0, j1):
                try:
                    val = obj.get(i, j)
                except Exception:
                    continue
                if val != 0:
                    out.set(i, j, float(val))

        _track_and_mark_temporary_if_native(out, deps=deps)
        return out
    except Exception:
        return None


def _try_convert_to_dense_f64(obj: Any, *, deps: OpsDeps) -> Any | None:
    """Materialize a symmetric/antisymmetric native matrix to a dense float64 matrix.

    The native matmul dispatch does not accept SymmetricMatrix/AntiSymmetricMatrix
    operands, so correctness-first matmul routes through a dense materialization.
    """
    dense_cls = getattr(deps.native, "FloatMatrix", None)
    if dense_cls is None:
        return None

    shape = _safe_rows_cols(obj)
    if shape is None:
        return None
    rows, cols = shape
    if rows != cols:
        return None

    try:
        out = dense_cls(rows)
        for i in range(rows):
            for j in range(cols):
                v = obj.get(i, j)
                if v != 0:
                    out.set(i, j, float(v))
        _track_and_mark_temporary_if_native(out, deps=deps)
        return out
    except Exception:
        return None


def _set_result_structure_properties(result: Any, *, structure: str) -> None:
    try:
        mapping: dict[str, Any] = {}
        if structure == "zero":
            mapping["is_zero"] = True
        elif structure == "identity":
            mapping["is_identity"] = True
        elif structure == "diagonal":
            mapping["is_diagonal"] = True
        elif structure == "upper_triangular":
            mapping["is_upper_triangular"] = True
        elif structure == "lower_triangular":
            mapping["is_lower_triangular"] = True
        _props.set_properties(result, mapping)
    except Exception:
        pass


def _matmul_result_structure(a_struct: str, b_struct: str) -> str:
    if a_struct == "zero" or b_struct == "zero":
        return "zero"
    if a_struct == "identity":
        return b_struct
    if b_struct == "identity":
        return a_struct

    if a_struct == "diagonal" and b_struct == "diagonal":
        return "diagonal"

    if a_struct == "diagonal" and b_struct in ("upper_triangular", "lower_triangular"):
        return b_struct
    if b_struct == "diagonal" and a_struct in ("upper_triangular", "lower_triangular"):
        return a_struct

    if a_struct == b_struct and a_struct in ("upper_triangular", "lower_triangular"):
        return a_struct

    return "general"


def matmul(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    # Materialize dtype-deferred allocations so they take the native fast path
    # (and produce the same promoted result as the `@` operator).
    try:
        from .lazy_allocation import LazyAllocated as _LazyAllocated
        from .lazy_allocation import _is_bit_matrix as _is_bit_matrix
    except Exception:  # pragma: no cover
        _LazyAllocated = None  # type: ignore[assignment]
        _is_bit_matrix = None  # type: ignore[assignment]

    if _LazyAllocated is not None:
        if isinstance(a, _LazyAllocated):
            a = a._materialize("bool" if _is_bit_matrix(b) else None)
        if isinstance(b, _LazyAllocated):
            b = b._materialize("bool" if _is_bit_matrix(a) else None)

    rec = _record_io_trace("matmul", [a, b], deps=deps)
    _prefetch_if_streaming(rec, [a, b], deps=deps)
    # Phase F integration: BlockMatrix routing.
    # If either operand is a BlockMatrix, preserve 'once block, always block'
    # by returning a thunked BlockMatrix via block orchestration.
    try:
        from .blockmatrix import BlockMatrix, block_matmul
    except Exception:  # pragma: no cover
        BlockMatrix = None  # type: ignore[assignment]
        block_matmul = None  # type: ignore[assignment]

    if BlockMatrix is not None and (isinstance(a, BlockMatrix) or isinstance(b, BlockMatrix)):
        if not isinstance(a, BlockMatrix):
            a = BlockMatrix([[a]])
        if not isinstance(b, BlockMatrix):
            b = BlockMatrix([[b]])
        return block_matmul(a, b)

    native_matmul = getattr(deps.native, "matmul", None)

    native_matrix_base = getattr(deps.native, "MatrixBase", None)
    native_vector_base = getattr(deps.native, "VectorBase", None)

    # Streaming-enforced path: if routed streaming and not blockmatrix, use tile-based matmul.
    try:
        if isinstance(rec, dict) and rec.get("route") == "streaming":
            streaming_res = _streaming_matmul_tiles(a, b, deps=deps, rec=rec)
            if streaming_res is not None:
                deps.track_matrix(streaming_res)
                deps.mark_temporary_if_auto(streaming_res)
                _discard_if_streaming(rec, [a, b], streaming_res, deps=deps)
                return streaming_res
    except Exception:
        pass

    # NumPy-like behavior: allow vectors in matmul by deferring to the native
    # operator implementation (which encodes the 1D rules).
    if native_vector_base is not None and (
        isinstance(a, native_vector_base) or isinstance(b, native_vector_base)
    ):
        result = a @ b
        if native_matrix_base is not None and isinstance(result, native_matrix_base):
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)
        elif native_vector_base is not None and isinstance(result, native_vector_base):
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)
        return result

    if native_matrix_base is not None:
        if isinstance(a, native_matrix_base) and isinstance(b, native_matrix_base):
            a_struct = _effective_structure_for(a)
            b_struct = _effective_structure_for(b)

            # Property-aware dispatch (Phase E): if users assert structure,
            # convert into the corresponding structured storage type so the
            # backend can take specialized paths.
            a_eff = a
            b_eff = b

            # Diagonal × Dense and Dense × Diagonal fast paths (float64 only).
            float_matrix = getattr(deps.native, "FloatMatrix", None)
            if a_struct == "diagonal" and float_matrix is not None and isinstance(b, float_matrix):
                converted = _try_convert_to_diagonal_f64(a, deps=deps)
                if converted is not None:
                    a_eff = converted
            elif b_struct == "diagonal" and float_matrix is not None and isinstance(a, float_matrix):
                converted = _try_convert_to_diagonal_f64(b, deps=deps)
                if converted is not None:
                    b_eff = converted

            # Triangular × Triangular fast path (float64 only).
            if a_struct in ("upper_triangular", "lower_triangular") and b_struct in (
                "upper_triangular",
                "lower_triangular",
            ):
                a_tri = _try_convert_to_triangular_f64(a, which=a_struct, deps=deps)
                b_tri = _try_convert_to_triangular_f64(b, which=b_struct, deps=deps)
                if a_tri is not None and b_tri is not None:
                    a_eff = a_tri
                    b_eff = b_tri

            # Symmetric/AntiSymmetric -> dense float64 fallback: the native matmul
            # dispatch does not accept these structured types, so materialize them
            # to dense before multiplying (correctness-first; the result is dense).
            if a_struct in ("symmetric", "antisymmetric"):
                converted = _try_convert_to_dense_f64(a, deps=deps)
                if converted is not None:
                    a_eff = converted
                    a_struct = "general"
            if b_struct in ("symmetric", "antisymmetric"):
                converted = _try_convert_to_dense_f64(b, deps=deps)
                if converted is not None:
                    b_eff = converted
                    b_struct = "general"

            # Prefer the native @ operator (MatrixBase.__matmul__), which is the
            # most widely supported entry point across matrix types.
            try:
                result = a_eff @ b_eff
            except Exception:
                # Some builds may still expose native.matmul for specific types.
                if native_matmul is None:
                    raise
                result = native_matmul(a_eff, b_eff)
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)

            _set_result_structure_properties(
                result, structure=_matmul_result_structure(a_struct, b_struct)
            )
            _discard_if_streaming(rec, [a, b], result, deps=deps)
            return result

    # Generic fallback
    if not (hasattr(a, "shape") and hasattr(b, "shape")):
        raise TypeError("Inputs must be matrix-like objects with a shape property.")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    rows = a.shape[0]
    cols = b.shape[1]
    inner = a.shape[1]

    np_module = deps.np_module
    if np_module is not None:
        try:
            a_np = np_module.array(
                [[a.get(i, j) for j in range(a.shape[1])] for i in range(a.shape[0])]
            )
            b_np = np_module.array(
                [[b.get(i, j) for j in range(b.shape[1])] for i in range(b.shape[0])]
            )
            res_np = np_module.matmul(a_np, b_np)
            res_mat = deps.Matrix(res_np)
            _discard_if_streaming(rec, [a, b], res_mat, deps=deps)
            return res_mat
        except Exception:
            pass

    # Slow generic loop (materializes the result in memory).
    res_data: list[list[Any]] = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val: Any = 0
            for k in range(inner):
                val += a.get(i, k) * b.get(k, j)
            res_data[i][j] = val
    res_mat = deps.Matrix(res_data)
    _discard_if_streaming(rec, [a, b], res_mat, deps=deps)
    return res_mat


def compute_k(matrix: Any, a: float, *, deps: OpsDeps) -> Any:
    func = getattr(deps.native, "compute_k_matrix", None)
    if func is None:
        raise NotImplementedError("compute_k_matrix is not available in this build.")

    result = func(matrix, a, 0)
    deps.track_matrix(result)
    deps.mark_temporary_if_auto(result)
    return result


def bitwise_not(matrix: Any, *, deps: OpsDeps) -> Any:
    if hasattr(matrix, "__invert__"):
        return ~matrix

    np_module = deps.np_module
    if np_module is not None:
        try:
            return np_module.invert(matrix)
        except Exception:
            pass

    raise TypeError("Object does not support bitwise inversion.")


def _bitwise_binop(a: Any, b: Any, kind: str, *, deps: OpsDeps) -> Any:
    """Bitwise AND/OR/XOR that always returns a bit structure."""
    op_name = {"and": "__and__", "or": "__or__", "xor": "__xor__"}[kind]
    fn = getattr(a, op_name, None)
    if callable(fn):
        try:
            result = fn(b)
            _track_and_mark_temporary_if_native(result, deps=deps)
            return result
        except Exception:
            pass

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError(f"bitwise_{kind} requires NumPy")
    a_np = np_module.asarray(_to_numpy_matrix(a, deps=deps)).astype(bool)
    b_np = np_module.asarray(_to_numpy_matrix(b, deps=deps)).astype(bool)
    fn_np = {"and": np_module.bitwise_and, "or": np_module.bitwise_or,
             "xor": np_module.bitwise_xor}[kind]
    result = fn_np(a_np, b_np)
    if result.ndim == 1:
        return _as_pycauset_vector(result, deps=deps)
    return _as_pycauset_array(result, deps=deps)


def bitwise_and(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise AND. Always returns a bit matrix or bit vector."""
    _record_io_trace("bitwise_and", [a, b], deps=deps)
    return _bitwise_binop(a, b, "and", deps=deps)


def bitwise_or(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise OR. Always returns a bit matrix or bit vector."""
    _record_io_trace("bitwise_or", [a, b], deps=deps)
    return _bitwise_binop(a, b, "or", deps=deps)


def bitwise_xor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise XOR. Always returns a bit matrix or bit vector."""
    _record_io_trace("bitwise_xor", [a, b], deps=deps)
    return _bitwise_binop(a, b, "xor", deps=deps)


def bitwise_nand(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NAND (NOT AND)."""
    _record_io_trace("bitwise_nand", [a, b], deps=deps)
    return bitwise_not(bitwise_and(a, b, deps=deps), deps=deps)


def bitwise_nor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NOR (NOT OR)."""
    _record_io_trace("bitwise_nor", [a, b], deps=deps)
    return bitwise_not(bitwise_or(a, b, deps=deps), deps=deps)


def bitwise_xnor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise XNOR (NOT XOR)."""
    _record_io_trace("bitwise_xnor", [a, b], deps=deps)
    return bitwise_not(bitwise_xor(a, b, deps=deps), deps=deps)


def invert(matrix: Any, *, deps: OpsDeps) -> Any:
    rec = _record_io_trace("invert", [matrix], deps=deps)
    _prefetch_if_streaming(rec, [matrix], deps=deps)
    try:
        if isinstance(rec, dict) and rec.get("route") == "streaming":
            streaming_res = _streaming_invert(matrix, deps=deps, rec=rec)
            if streaming_res is not None:
                _discard_if_streaming(rec, [matrix], streaming_res, deps=deps)
                return streaming_res
    except Exception:
        pass
    native_exc: Exception | None = None
    if hasattr(matrix, "invert"):
        try:
            result = matrix.invert()
        except Exception as exc:
            native_exc = exc
        else:
            _track_and_mark_temporary_if_native(result, deps=deps)
            _discard_if_streaming(rec, [matrix], result, deps=deps)
            return result

    np_module = deps.np_module
    if np_module is not None:
        try:
            result = np_module.linalg.inv(matrix)
            result_arr = _as_pycauset_array(result, deps=deps)
            _discard_if_streaming(rec, [matrix], result_arr, deps=deps)
            return result_arr
        except Exception:
            if native_exc is not None:
                raise native_exc

    raise TypeError("Object does not support matrix inversion.")


def lstsq(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Return a least-squares solution x minimizing ||a @ x - b||.

    Baseline implementation uses normal equations: x = (A^T A)^{-1} A^T b.

    Notes:
    - This is intended as an endpoint-first baseline.
    - It can be numerically unstable compared to QR/SVD.
    """
    a_t = getattr(a, "T", None)
    if a_t is None:
        a_t = getattr(a, "transpose", None)
        if callable(a_t):
            a_t = a_t()
    if a_t is None:
        raise TypeError("lstsq: expected a matrix-like object with transpose support")

    ata = matmul(a_t, a, deps=deps)
    atb = matmul(a_t, b, deps=deps)
    result = solve(ata, atb, deps=deps)
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


def slogdet(a: Any, *, deps: OpsDeps) -> tuple[float, float]:
    """Return (sign, log(abs(det(a)))) for square matrices."""
    import math

    det_fn = getattr(a, "determinant", None)
    if callable(det_fn):
        det = float(det_fn())
    else:
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for slogdet")
        det = float(np_module.linalg.det(_to_numpy_matrix(a, deps=deps)))  # pragma: no cover

    if det == 0.0:
        return 0.0, float("-inf")

    sign = 1.0 if det > 0.0 else -1.0
    return sign, float(math.log(abs(det)))


def cond(a: Any, *, deps: OpsDeps, p: Any = None) -> float:
    """Compute a condition number estimate using ||A|| * ||A^{-1}||.

    Currently uses Frobenius norm for matrices (matches `pycauset.norm`).
    """
    if p is not None:
        raise NotImplementedError("cond(p=...) is not implemented; only default norm is supported")

    norm_fn = getattr(deps.native, "norm", None)
    if norm_fn is None:
        raise RuntimeError("native norm is not available")

    inv_a = invert(a, deps=deps)
    return float(norm_fn(a) * norm_fn(inv_a))


def eigh(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Eigen-decomposition for real symmetric / complex Hermitian matrices (native preferred)."""
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eigh requires a square matrix")
    rec = _record_io_trace("eigh", [a], deps=deps, supports_streaming=False)
    _prefetch_if_streaming(rec, [a], deps=deps)
    
    # Check Cache
    ctx_w = _try_load_eigen_cache(a, "eigenvalues", getattr(deps.native, "FloatVector", None), deps)
    
    # Eigenvectors: Real -> FloatMatrix, Complex -> ComplexFloat64Matrix
    vec_cls = getattr(deps.native, "FloatMatrix", None)
    complex_types = (
        getattr(deps.native, "ComplexFloat64Matrix", type(None)),
        getattr(deps.native, "ComplexFloat32Matrix", type(None)),
    )
    if isinstance(a, complex_types):
        vec_cls = getattr(deps.native, "ComplexFloat64Matrix", None)

    ctx_v = _try_load_eigen_cache(a, "eigenvectors", vec_cls, deps)
    
    if ctx_w is not None and ctx_v is not None:
        _discard_if_streaming(rec, [a], None, deps=deps)
        return ctx_w, ctx_v

    result_w = None
    result_v = None

    # Correctness-first (2026-08-24): native eigh (R1_CPU Phase 6) crashes; use NumPy.
    # NumPy Fallback
    if result_w is None:
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for eigh")
        w, v = np_module.linalg.eigh(_to_numpy_matrix(a, deps=deps))
        result_w = _as_pycauset_array(w, deps=deps)
        result_v = _as_pycauset_array(v, deps=deps)

    # Save to Cache
    try:
        if getattr(a, "get_backing_file", lambda: None)() and _linalg_cache._is_new_container(a.get_backing_file()): # Reuse helper
             view_sig = _big_blob_cache.compute_view_signature(a)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvalues", obj=result_w, view_signature=view_sig)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvectors", obj=result_v, view_signature=view_sig)
    except Exception:
        pass

    _discard_if_streaming(rec, [a], None, deps=deps)
    return result_w, result_v


def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eigvalsh requires a square matrix")
    rec = _record_io_trace("eigvalsh", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)
    try:
        if isinstance(rec, dict) and rec.get("route") == "streaming":
            streaming_res = _streaming_eigvalsh(a, deps=deps, rec=rec)
            if streaming_res is not None:
                _discard_if_streaming(rec, [a], streaming_res, deps=deps)
                return streaming_res
    except Exception:
        pass
    """Eigenvalues for real symmetric / complex Hermitian matrices.

    Phase E wiring:
    - If cached-derived `a.properties["eigenvalues"]` exists, prefer it.
    - If `is_hermitian` is explicitly False, reject.
    """

    props = None
    try:
        props = _props.get_properties(a)
    except Exception:
        props = None

    if props is not None:
        if props.get("is_hermitian") is False:
            raise ValueError("eigvalsh requires is_hermitian != False")

        # Preferred: Big Blob Cache (supports full vectors)
        # We try this first before the small property list
        ctx_w = _try_load_eigen_cache(a, "eigenvalues", getattr(deps.native, "FloatVector", None), deps)
        if ctx_w is not None:
             _discard_if_streaming(rec, [a], ctx_w, deps=deps)
             return ctx_w

        if "eigenvalues" in props:
            try:
                return _as_pycauset_array(props["eigenvalues"], deps=deps)
            except Exception:
                pass

    result_w = None

    # Correctness-first (2026-08-24): native eigvalsh (R1_CPU Phase 6) crashes; use NumPy.
    if result_w is None:
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for eigvalsh")
        w = np_module.linalg.eigvalsh(_to_numpy_matrix(a, deps=deps))
        result_w = _as_pycauset_array(w, deps=deps)

    # Save to Cache
    try:
        if getattr(a, "get_backing_file", lambda: None)() and _linalg_cache._is_new_container(a.get_backing_file()): # Reuse helper
             view_sig = _big_blob_cache.compute_view_signature(a)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvalues", obj=result_w, view_signature=view_sig)
    except Exception:
        pass

    if props is not None:
        try:
            # Also populate legacy small cache if possible/reasonable? 
            # Actually, let's keep it for compatibility if it's small enough, but maybe not worth specific logic here.
            # We skip writing to props["eigenvalues"] to encourage big blob usage.
            pass
        except Exception:
            pass

    _discard_if_streaming(rec, [a], result_w, deps=deps)
    return result_w



def _try_load_eigen_cache(a: Any, name: str, cls: Any, deps: OpsDeps) -> Any | None:
    try:
        backing = getattr(a, "get_backing_file", lambda: None)()
        if not backing:
            return None
        
        # Must be a .pycauset container to support big_blob_cache
        if not backing.endswith(".pycauset") or not os.path.exists(backing):
            return None

        view_sig = _big_blob_cache.compute_view_signature(a)
        
        if cls is None: 
             return None

        res = _big_blob_cache.try_load_cached_matrix(
            backing,
            name=name,
            view_signature=view_sig,
            MatrixClass=cls
        )
        if res is not None:
             _track_and_mark_temporary_if_native(res, deps=deps)
        return res
    except Exception:
        return None


def eig(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Eigen-decomposition for general matrices (native preferred, NumPy fallback)."""
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eig requires a square matrix")
    rec = _record_io_trace("eig", [a], deps=deps, supports_streaming=False)
    _prefetch_if_streaming(rec, [a], deps=deps)
    
    # Check Cache
    ctx_w = _try_load_eigen_cache(a, "eigenvalues", getattr(deps.native, "ComplexFloat64Vector", None), deps)
    ctx_v = _try_load_eigen_cache(a, "eigenvectors", getattr(deps.native, "ComplexFloat64Matrix", None), deps)
    
    if ctx_w is not None and ctx_v is not None:
        _discard_if_streaming(rec, [a], None, deps=deps)
        return ctx_w, ctx_v

    result_w = None
    result_v = None

    # Correctness-first (2026-08-24): native eig (R1_CPU Phase 6) crashes; use NumPy.
    # NumPy Fallback
    if result_w is None:
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for eig")

        w, v = np_module.linalg.eig(_to_numpy_matrix(a, deps=deps))
        result_w = _as_pycauset_vector(w, deps=deps)
        result_v = _as_pycauset_array(v, deps=deps)

    # Save to Cache
    try:
        if getattr(a, "get_backing_file", lambda: None)() and _linalg_cache._is_new_container(a.get_backing_file()): # Reuse helper
             view_sig = _big_blob_cache.compute_view_signature(a)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvalues", obj=result_w, view_signature=view_sig)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvectors", obj=result_v, view_signature=view_sig)
    except Exception:
        pass

    _discard_if_streaming(rec, [a], None, deps=deps)
    return result_w, result_v


def eigvals(a: Any, *, deps: OpsDeps) -> Any:
    """Eigenvalues for general matrices (native preferred, NumPy fallback)."""
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eigvals requires a square matrix")
    rec = _record_io_trace("eigvals", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)

    # Check Cache
    ctx_w = _try_load_eigen_cache(a, "eigenvalues", getattr(deps.native, "ComplexFloat64Vector", None), deps)
    if ctx_w is not None:
         _discard_if_streaming(rec, [a], ctx_w, deps=deps)
         return ctx_w

    result_w = None

    # Correctness-first (2026-08-24): native eigvals (R1_CPU Phase 6) crashes; use NumPy.
    # NumPy Fallback
    if result_w is None:
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for eigvals")

        w = np_module.linalg.eigvals(_to_numpy_matrix(a, deps=deps))
        result_w = _as_pycauset_vector(w, deps=deps)

    # Save to Cache
    try:
         if getattr(a, "get_backing_file", lambda: None)() and _linalg_cache._is_new_container(a.get_backing_file()):
             view_sig = _big_blob_cache.compute_view_signature(a)
             _big_blob_cache.persist_cached_object(a.get_backing_file(), name="eigenvalues", obj=result_w, view_signature=view_sig)
    except Exception:
        pass

    _discard_if_streaming(rec, [a], result_w, deps=deps)
    return result_w

    w = np_module.linalg.eigvals(_to_numpy_matrix(a, deps=deps))
    w_out = _as_pycauset_array(w, deps=deps)
    _discard_if_streaming(rec, [a], w_out, deps=deps)
    return w_out


def eigvals_arnoldi(a: Any, k: int, m: int, tol: float, *, deps: OpsDeps) -> Any:
    """Top-k eigenvalues via Arnoldi/Lanczos-style iteration (when available).

    - Prefers native `eigvals_arnoldi` when provided by the extension.
    - Falls back to NumPy eigvals and returns the top-|k| by magnitude.
    - Records IO observability trace for parity with other eigen ops.
    """

    rec = _record_io_trace("eigvals_arnoldi", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)

    try:
        if isinstance(rec, dict) and rec.get("route") == "streaming":
            streaming_res = _streaming_eigvals_arnoldi(a, k, m, tol, deps=deps, rec=rec)
            if streaming_res is not None:
                _discard_if_streaming(rec, [a], streaming_res, deps=deps)
                return streaming_res
    except Exception:
        pass

    # Correctness-first (2026-08-24): the native eigvals_arnoldi (R1_CPU Phase 6)
    # crashes with an access violation, so use the NumPy fallback until fixed.
    np_module = deps.np_module
    if np_module is None:
        raise NotImplementedError("eigvals_arnoldi is not available (no native/NumPy fallback)")

    eigs = np_module.linalg.eigvals(_to_numpy_matrix(a, deps=deps))
    eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
    top = np_module.array(eigs_sorted[:k])
    out = _as_pycauset_vector(top, deps=deps)
    _discard_if_streaming(rec, [a], out, deps=deps)
    return out


def eigvals_skew(a: Any, k: int, *, deps: OpsDeps) -> Any:
    """Top-k (by magnitude) eigenvalues of a real skew-symmetric matrix.

    A real skew-symmetric matrix (A == -A.T) has purely imaginary eigenvalues
    that come in +/-i*lambda pairs (plus a zero eigenvalue for odd dimension).
    Prefers the native `eigvals_skew` when available; otherwise falls back to
    NumPy's general eigensolver and returns the top-|k| by magnitude.
    """

    if k <= 0:
        raise ValueError("eigvals_skew: k must be positive")
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eigvals_skew requires a square matrix")

    native_fn = getattr(deps.native, "eigvals_skew", None)
    if callable(native_fn):
        try:
            result = native_fn(a, k)
            _track_and_mark_temporary_if_native(result, deps=deps)
            return result
        except Exception:
            pass

    np_module = deps.np_module
    if np_module is None:
        raise NotImplementedError("eigvals_skew is not available (no native/NumPy fallback)")

    eigs = np_module.linalg.eigvals(_to_numpy_matrix(a, deps=deps))
    eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
    top = np_module.array(eigs_sorted[:k])
    return _as_pycauset_vector(top, deps=deps)


def solve_triangular(*_args: Any, **_kwargs: Any) -> Any:
    """Solve a triangular system using gospel properties.

    This endpoint exists primarily to enable Phase E property-aware shortcuts.
    Current implementation:
    - Diagonal: elementwise divide.
    - Upper/lower triangular: convert to TriangularFloatMatrix and use native triangular inversion.
    """

    if len(_args) < 2:
        raise TypeError("solve_triangular(a, b) requires two positional arguments")

    a = _args[0]
    b = _args[1]
    deps = _kwargs.get("deps")
    if deps is None:
        raise TypeError("solve_triangular requires deps")

    a_struct = _effective_structure_for(a)
    shape = _safe_rows_cols(a)
    if shape is None:
        raise TypeError("solve_triangular: expected a matrix-like object")
    n, m = shape
    if n != m:
        raise ValueError("solve_triangular: a must be square")

    # Diagonal fast path.
    if a_struct == "diagonal":
        diag = [float(a.get(i, i)) for i in range(n)]
        # Vector RHS
        native_vector_base = getattr(deps.native, "VectorBase", None)
        if native_vector_base is not None and isinstance(b, native_vector_base):
            if int(getattr(b, "size")()) != n:
                raise ValueError("solve_triangular: shape mismatch")
            try:
                data = [float(b.get(i)) / diag[i] for i in range(n)]
                return _as_pycauset_array(data, deps=deps)
            except Exception:
                pass

        # Matrix RHS (materialize to numpy for simplicity)
        np_module = deps.np_module
        if np_module is None:
            raise RuntimeError("NumPy is required for diagonal solve")
        b_np = np_module.asarray(b)
        x_np = b_np / np_module.asarray(diag).reshape((n, 1))
        return _as_pycauset_array(x_np, deps=deps)

    # Triangular path: use native triangular inversion (float64).
    if a_struct not in ("upper_triangular", "lower_triangular"):
        # If no structure claim is present, refuse (caller should use solve()).
        raise ValueError("solve_triangular: a is not marked triangular")

    a_tri = _try_convert_to_triangular_f64(a, which=a_struct, deps=deps)
    if a_tri is None:
        raise RuntimeError("solve_triangular: triangular conversion unavailable")

    inv_a = invert(a_tri, deps=deps)
    result = matmul(inv_a, b, deps=deps)
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


def pinv(a: Any, *, deps: OpsDeps) -> Any:
    """Moore-Penrose pseudoinverse (baseline).

    Uses the normal equations:
    - tall/square (`rows >= cols`): (A^T A)^-1 A^T
    - wide (`rows < cols`):          A^T (A A^T)^-1

    This matches the `lstsq` baseline (normal equations) and is numerically less
    stable than an SVD-based pinv. Falls back to NumPy's SVD-based `pinv` when the
    normal equations fail (e.g. rank-deficient) or for non-native inputs.
    """
    _record_io_trace("pinv", [a], deps=deps)

    shape = _safe_rows_cols(a)
    if shape is not None:
        rows, cols = shape
        a_t = getattr(a, "T", None)
        if a_t is None:
            a_t = getattr(a, "transpose", None)
            if callable(a_t):
                a_t = a_t()
        if a_t is not None:
            try:
                if rows >= cols:
                    ata = matmul(a_t, a, deps=deps)
                    ata_inv = invert(ata, deps=deps)
                    result = matmul(ata_inv, a_t, deps=deps)
                else:
                    aat = matmul(a, a_t, deps=deps)
                    aat_inv = invert(aat, deps=deps)
                    result = matmul(a_t, aat_inv, deps=deps)
                _track_and_mark_temporary_if_native(result, deps=deps)
                return result
            except Exception:
                pass

    # NumPy fallback (SVD-based; also covers complex and non-native inputs).
    np_module = deps.np_module
    if np_module is not None:
        val = np_module.linalg.pinv(_to_numpy_matrix(a, deps=deps))
        return _as_pycauset_array(val, deps=deps)

    raise RuntimeError("pinv failed")


def trace(a: Any, *, deps: OpsDeps) -> Any:
    """Return the sum of the diagonal elements."""
    rec = _record_io_trace("trace", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)

    # Native method or property
    fn = getattr(a, "trace", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass

    # NumPy Fallback
    np_module = deps.np_module
    if np_module:
        return np_module.trace(_to_numpy_matrix(a, deps=deps))
    return 0.0


def determinant(a: Any, *, deps: OpsDeps) -> Any:
    """Return the determinant of a square matrix."""
    rec = _record_io_trace("determinant", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)

    # Native method
    fn = getattr(a, "determinant", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass

    # NumPy Fallback
    np_module = deps.np_module
    if np_module:
        return np_module.linalg.det(_to_numpy_matrix(a, deps=deps))
    return 0.0


def norm(x: Any, ord: Any = None, *, deps: OpsDeps) -> float:
    """Matrix or vector norm.

    ord=None or 'fro': Frobenius norm (matrix) / L2 norm (vector), native.
    ord=2: spectral norm (matrix, largest singular value) / L2 norm (vector).
    Other ord values (1, inf, 'nuc', ...) use the NumPy fallback.
    """
    rec = _record_io_trace("norm", [x], deps=deps)
    _prefetch_if_streaming(rec, [x], deps=deps)

    # Structural shortcuts for the default (Frobenius) norm.
    x_struct = _effective_structure_for(x)
    shape = _safe_rows_cols(x)
    if ord is None or ord == 'fro':
        if x_struct == "zero":
            return 0.0
        if x_struct == "identity" and shape is not None and shape[0] > 0:
            return float(min(shape) ** 0.5)

    # Native norm only for the default Frobenius/L2 case. For ord=2 on a matrix,
    # the native function computes Frobenius, not the spectral norm, so route
    # ord=2 and every other ord through NumPy (which is correct).
    if ord is None or ord == 'fro':
        fn = getattr(deps.native, "norm", None)
        if callable(fn):
            try:
                return float(fn(x))
            except Exception:
                pass

    # NumPy Fallback
    np_module = deps.np_module
    if np_module:
        return float(np_module.linalg.norm(_to_numpy_matrix(x, deps=deps), ord=ord))
    return 0.0


def svdvals(a: Any, *, deps: OpsDeps) -> Any:
    """Singular values of a matrix (the S vector of an SVD), descending."""
    _record_io_trace("svdvals", [a], deps=deps)

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("svdvals requires NumPy")
    s = np_module.linalg.svd(_to_numpy_matrix(a, deps=deps), compute_uv=False)
    return _as_pycauset_vector(s, deps=deps)


def matrix_rank(a: Any, tol: Any = None, *, deps: OpsDeps) -> int:
    """Numerical rank of a matrix (number of singular values above `tol`).

    Structural shortcuts avoid an SVD: zero -> 0, identity -> min(m, n), and
    diagonal/triangular -> count of non-zero diagonal entries.
    """
    _record_io_trace("matrix_rank", [a], deps=deps)

    a_struct = _effective_structure_for(a)
    shape = _safe_rows_cols(a)
    if a_struct == "zero":
        return 0
    if a_struct == "identity":
        return min(shape) if shape else 0
    if a_struct in ("diagonal", "upper_triangular", "lower_triangular"):
        n = min(shape) if shape else 0
        non_zero = 0
        for i in range(n):
            try:
                if a.get(i, i) != 0:
                    non_zero += 1
            except Exception:
                pass
        return non_zero

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("matrix_rank requires NumPy")
    return int(np_module.linalg.matrix_rank(_to_numpy_matrix(a, deps=deps), tol=tol))


def matrix_power(a: Any, n: int, *, deps: OpsDeps) -> Any:
    """Integer power of a square matrix (A^n) via binary exponentiation.

    Structural shortcuts: identity stays identity, zero stays zero (n > 0), and a
    diagonal matrix is raised elementwise.
    """
    _record_io_trace("matrix_power", [a], deps=deps)
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("matrix_power requires a square matrix")

    if n == 0:
        return _identity_like(a, shape, deps)
    if n == 1:
        return a

    a_struct = _effective_structure_for(a)
    if a_struct == "identity":
        return a if n >= 0 else a
    if a_struct == "zero":
        if n > 0:
            return a
        raise ValueError("matrix_power: zero matrix has no negative power")

    if n < 0:
        inv = invert(a, deps=deps)
        return matrix_power(inv, -n, deps=deps)

    # Binary exponentiation via matmul.
    result = _identity_like(a, shape, deps)
    base = a
    while n > 0:
        if n & 1:
            result = matmul(result, base, deps=deps)
        base = matmul(base, base, deps=deps)
        n >>= 1
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


def _identity_like(a: Any, shape: Any, deps: OpsDeps) -> Any:
    """Return an identity matrix matching a's shape and dtype (best effort)."""
    I_cls = getattr(deps.native, "IdentityMatrix", None)
    if I_cls is not None and shape is not None:
        try:
            return I_cls(shape[0])
        except Exception:
            pass
    np_module = deps.np_module
    if np_module is not None and shape is not None:
        return _as_pycauset_array(np_module.eye(shape[0], shape[1]), deps=deps)
    raise RuntimeError("matrix_power: cannot build identity matrix")


def outer(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Outer product of two vectors: out[i, j] = a[i] * b[j]."""
    _record_io_trace("outer", [a, b], deps=deps)

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("outer requires NumPy")
    a_np = np_module.asarray(_to_numpy_matrix(a, deps=deps)).ravel()
    b_np = np_module.asarray(_to_numpy_matrix(b, deps=deps)).ravel()
    return _as_pycauset_array(np_module.outer(a_np, b_np), deps=deps)


def cross(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Cross product of two 3-element vectors."""
    _record_io_trace("cross", [a, b], deps=deps)

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("cross requires NumPy")
    a_np = np_module.asarray(_to_numpy_matrix(a, deps=deps)).ravel()
    b_np = np_module.asarray(_to_numpy_matrix(b, deps=deps)).ravel()
    if a_np.size != 3 or b_np.size != 3:
        raise ValueError("cross product requires vectors of length 3")
    return _as_pycauset_vector(np_module.cross(a_np, b_np), deps=deps)


def vecdot(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Conjugate dot product: sum(conj(a) * b). For real inputs this equals dot."""
    _record_io_trace("vecdot", [a, b], deps=deps)

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("vecdot requires NumPy")
    a_np = np_module.asarray(_to_numpy_matrix(a, deps=deps)).ravel()
    b_np = np_module.asarray(_to_numpy_matrix(b, deps=deps)).ravel()
    result = np_module.vdot(a_np, b_np)
    if np_module.iscomplexobj(result):
        return complex(result)
    return float(result)


def cholesky(a: Any, *, deps: OpsDeps) -> Any:
    """Return the Cholesky decomposition."""
    rec = _record_io_trace("cholesky", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)

    fn = getattr(deps.native, "cholesky", None)
    if callable(fn):
        try:
            out = fn(a)
            _track_and_mark_temporary_if_native(out, deps=deps)
            return out
        except Exception:
            pass
            
    # NumPy Fallback
    np_module = deps.np_module
    if np_module:
        val = np_module.linalg.cholesky(_to_numpy_matrix(a, deps=deps))
        return _as_pycauset_array(val, deps=deps)
    raise RuntimeError("cholesky failed")


def qr(a: Any, mode: str = 'reduced', *, deps: OpsDeps) -> Any:
    """Return QR decomposition."""
    rec = _record_io_trace("qr", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)
    
    # Native default is reduced
    if mode == 'reduced':
        fn = getattr(deps.native, "qr", None)
        if callable(fn):
            try:
                q, r = fn(a)
                _track_and_mark_temporary_if_native(q, deps=deps)
                _track_and_mark_temporary_if_native(r, deps=deps)
                return q, r
            except Exception:
                pass

    np_module = deps.np_module
    if np_module:
        q_np, r_np = np_module.linalg.qr(_to_numpy_matrix(a, deps=deps), mode=mode)
        q = _as_pycauset_array(q_np, deps=deps)
        r = _as_pycauset_array(r_np, deps=deps)
        return q, r
    raise RuntimeError("qr failed")


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, *, deps: OpsDeps) -> Any:
    """Return SVD decomposition."""
    _record_io_trace("svd", [a], deps=deps)
    
    # Native only supports reduced/compact where U is MxK, VT is KxN (approx full_matrices=False)
    if not full_matrices and compute_uv:
        fn = getattr(deps.native, "svd", None)
        if callable(fn):
            try:
                u, s, vt = fn(a)
                _track_and_mark_temporary_if_native(u, deps=deps)
                _track_and_mark_temporary_if_native(s, deps=deps)
                _track_and_mark_temporary_if_native(vt, deps=deps)
                return u, s, vt
            except Exception:
                pass

    np_module = deps.np_module
    if np_module:
        res = np_module.linalg.svd(_to_numpy_matrix(a, deps=deps), full_matrices=full_matrices, compute_uv=compute_uv)
        if compute_uv:
             u_np, s_np, vt_np = res
             return (_as_pycauset_array(u_np, deps=deps), _as_pycauset_array(s_np, deps=deps), _as_pycauset_array(vt_np, deps=deps))
        else:
             s_np = res
             return _as_pycauset_array(s_np, deps=deps)
        
    raise RuntimeError("svd failed")


def lu(a: Any, *, deps: OpsDeps) -> Any:
    """Return LU decomposition (P, L, U)."""
    _record_io_trace("lu", [a], deps=deps)
    
    fn = getattr(deps.native, "lu", None)
    if callable(fn):
        p, l_mat, u = fn(a)
        _track_and_mark_temporary_if_native(p, deps=deps)
        _track_and_mark_temporary_if_native(l_mat, deps=deps)
        _track_and_mark_temporary_if_native(u, deps=deps)
        return p, l_mat, u

    # Scipy Fallback is not standard here as scipy is not in standard deps
    raise NotImplementedError("lu requires native implementation")


def solve(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve AX = B, honouring properties-as-gospel (identity/zero/triangular)."""
    _record_io_trace("solve", [a, b], deps=deps)

    a_struct = _effective_structure_for(a)
    shape = _safe_rows_cols(a)

    if a_struct == "zero":
        raise ValueError("solve: matrix marked is_zero; system is singular")

    if a_struct == "identity":
        # Treat as identity regardless of payload; check basic shape compatibility.
        if shape is not None and shape[0] != shape[1]:
            raise ValueError("solve: is_identity requires a square matrix for solve")
        native_matrix_base = getattr(deps.native, "MatrixBase", None)
        native_vector_base = getattr(deps.native, "VectorBase", None)
        if (native_matrix_base and isinstance(b, native_matrix_base)) or (
            native_vector_base and isinstance(b, native_vector_base)
        ):
            _track_and_mark_temporary_if_native(b, deps=deps)
            return b
        return _as_pycauset_array(b, deps=deps)

    if a_struct in ("upper_triangular", "lower_triangular", "diagonal"):
        try:
            return solve_triangular(a, b, deps=deps)
        except Exception:
            pass

    fn = getattr(deps.native, "solve", None)
    if callable(fn):
        try:
            x = fn(a, b)
            _track_and_mark_temporary_if_native(x, deps=deps)
            return x
        except Exception:
            pass

    np_module = deps.np_module
    if np_module:
        val = np_module.linalg.solve(_to_numpy_matrix(a, deps=deps), _to_numpy_matrix(b, deps=deps))
        return _as_pycauset_array(val, deps=deps)

    raise RuntimeError("solve failed")
