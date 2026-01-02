from __future__ import annotations

from typing import Any

from . import properties as _props
from . import export_guard
from . import io_observability


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
    try:
        props = _props.get_properties(obj)
        return _props.effective_structure_from_properties(props)
    except Exception:
        return "general"


def _record_io_trace(op_name: str, operands: list[Any], *, deps: OpsDeps, allow_huge: bool = False) -> None:
    manager = getattr(deps, "streaming_manager", None)
    if manager is not None:
        try:
            return manager.plan(op_name, operands, allow_huge=allow_huge)
        except Exception:
            pass

    observer = getattr(deps, "io_observer", None)
    if observer is None:
        return None
    try:
        return observer.plan_and_record(op_name, operands, allow_huge=allow_huge)
    except Exception:
        return None


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


def solve(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve a linear system a @ x = b.

    Baseline implementation:
    - If the matrix provides `.solve(b)`, use it.
    - Otherwise, compute `invert(a) @ b`.
    """
    fn = getattr(a, "solve", None)
    if callable(fn):
        result = fn(b)
        _track_and_mark_temporary_if_native(result, deps=deps)
        return result

    a_struct = _effective_structure_for(a)
    shape = _safe_rows_cols(a)

    if a_struct == "zero":
        raise ValueError("solve: matrix marked is_zero; system is singular")

    if a_struct == "identity":
        # Treat as identity regardless of payload; check basic shape compatibility when known.
        if shape is not None and shape[0] != shape[1]:
            raise ValueError("solve: is_identity requires a square matrix for solve")

        # If RHS is native, return it directly; otherwise coerce to a native array.
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
            # Fallback to generic path.
            pass

    inv_a = invert(a, deps=deps)
    result = matmul(inv_a, b, deps=deps)
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


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
    rec = _record_io_trace("eigh", [a], deps=deps)
    _prefetch_if_streaming(rec, [a], deps=deps)
    try:
        if isinstance(rec, dict) and rec.get("route") == "streaming":
            streaming_res = _streaming_eigh(a, deps=deps, rec=rec)
            if streaming_res is not None:
                _discard_if_streaming(rec, [a], None, deps=deps)
                return streaming_res
    except Exception:
        pass
    """Eigen-decomposition for real symmetric / complex Hermitian matrices (NumPy fallback)."""
    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for eigh")
    w, v = np_module.linalg.eigh(_to_numpy_matrix(a, deps=deps))
    w_out = _as_pycauset_array(w, deps=deps)
    v_out = _as_pycauset_array(v, deps=deps)
    _discard_if_streaming(rec, [a], None, deps=deps)
    return w_out, v_out


def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
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

        if "eigenvalues" in props:
            try:
                return _as_pycauset_array(props["eigenvalues"], deps=deps)
            except Exception:
                pass

    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for eigvalsh")
    w = np_module.linalg.eigvalsh(_to_numpy_matrix(a, deps=deps))
    out = _as_pycauset_array(w, deps=deps)

    if props is not None:
        try:
            props["eigenvalues"] = [complex(x).real if complex(x).imag == 0 else complex(x) for x in w.tolist()]
        except Exception:
            pass

            _discard_if_streaming(rec, [a], out, deps=deps)
    return out


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

    fn = getattr(deps.native, "eigvals_arnoldi", None)
    if callable(fn):
        try:
            result = fn(a, k, m, tol)
            _track_and_mark_temporary_if_native(result, deps=deps)
            _discard_if_streaming(rec, [a], result, deps=deps)
            return result
        except Exception:
            pass

    np_module = deps.np_module
    if np_module is None:
        raise NotImplementedError("eigvals_arnoldi is not available (no native/NumPy fallback)")

    eigs = np_module.linalg.eigvals(_to_numpy_matrix(a, deps=deps))
    eigs_sorted = sorted(eigs, key=lambda x: abs(x), reverse=True)
    top = np_module.array(eigs_sorted[:k])
    out = _as_pycauset_array(top, deps=deps)
    _discard_if_streaming(rec, [a], out, deps=deps)
    return out


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


def lu(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("lu is not available yet")


def cholesky(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("cholesky is not available yet")


def svd(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("svd is not available yet")


def pinv(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("pinv is not available yet")
