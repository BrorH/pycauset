from __future__ import annotations

import os
from typing import Any

from . import export_guard, io_observability
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
    (e.g. general eigenvalues) is copied into a complex_float64 DenseVector.
    """
    np_module = deps.np_module
    if np_module is not None and np_module.iscomplexobj(obj):
        flat = np_module.asarray(obj, dtype=np_module.complex128).reshape(-1)
        out = deps.native.DenseVector(int(flat.shape[0]), dtype="complex_float64")
        for i, z in enumerate(flat.tolist()):
            out[i] = z
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
        track_matrix: Any,
        mark_temporary_if_auto: Any,
        warnings_module: Any,
        io_observer: Any | None,
        streaming_manager: Any | None = None,
    ) -> None:
        self.native = native
        self.np_module = np_module
        self.Matrix = Matrix
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

            # The engine dispatches every structure and dtype itself. The conversions that
            # were here (symmetric -> dense float64, triangular/diagonal -> float64) changed
            # result dtypes and duplicated engine paths (F-073).
            result = a @ b
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)

            _set_result_structure_properties(
                result, structure=_matmul_result_structure(a_struct, b_struct)
            )
            _discard_if_streaming(rec, [a, b], result, deps=deps)
            return result

    # Array-likes (NumPy arrays, nested lists): import, then the engine. No per-element or
    # NumPy fallback (R1: NumPy is an oracle, not a backend).
    if not isinstance(a, (native_matrix_base, native_vector_base)) or not isinstance(
        b, (native_matrix_base, native_vector_base)
    ):
        return matmul(_import_operand(a, deps=deps), _import_operand(b, deps=deps), deps=deps)
    raise TypeError(f"matmul: unsupported operands {type(a).__name__}, {type(b).__name__}")


def _import_operand(x: Any, *, deps: OpsDeps) -> Any:
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    if isinstance(x, native):
        return x
    arr = deps.np_module.asarray(x)
    if arr.ndim == 1:
        return deps.native.asarray(arr)
    return deps.Matrix(arr)


def compute_k(matrix: Any, a: float, *, deps: OpsDeps) -> Any:
    func = getattr(deps.native, "compute_k_matrix", None)
    if func is None:
        raise NotImplementedError("compute_k_matrix is not available in this build.")

    result = func(matrix, a, 0)
    deps.track_matrix(result)
    deps.mark_temporary_if_auto(result)
    return result


def bitwise_not(matrix: Any, *, deps: OpsDeps) -> Any:
    """~x (NumPy invert): logical not for bool, bitwise not for integers. Engine only."""
    return ~matrix


def bitwise_and(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a & b with NumPy's dtype rules (bool & bool is bool; integers promote). Engine only."""
    _record_io_trace("bitwise_and", [a, b], deps=deps)
    return a & b


def bitwise_or(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a | b with NumPy's dtype rules. Engine only."""
    _record_io_trace("bitwise_or", [a, b], deps=deps)
    return a | b


def bitwise_xor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a ^ b with NumPy's dtype rules. Engine only."""
    _record_io_trace("bitwise_xor", [a, b], deps=deps)
    return a ^ b


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
    if not isinstance(matrix, deps.native.MatrixBase):
        raise TypeError("Object does not support matrix inversion.")
    # The engine inverts every dtype, structure and view (LinearAlgebra.cpp `inverse`); a
    # singular matrix raises. No NumPy fallback.
    result = matrix.invert()
    _track_and_mark_temporary_if_native(result, deps=deps)
    _discard_if_streaming(rec, [matrix], result, deps=deps)
    return result


def lstsq(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Least-squares solution x minimizing ||a @ x - b|| (minimum norm when rank-deficient).

    x = pinv(a) @ b, through the SVD: the same answer as NumPy's lstsq (which uses ?gelsd).
    The normal equations (A^T A)^-1 A^T b used before squared the condition number and used
    A^T instead of A^H for complex input (F-069).
    """
    result = matmul(pinv(a, deps=deps), b, deps=deps)
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


def slogdet(a: Any, *, deps: OpsDeps) -> tuple[Any, float]:
    """(sign, log|det|) from the engine's LU pivots, as NumPy: no overflow for large n, a
    complex unit sign for complex matrices, (0, -inf) when singular (F-070)."""
    return a.slogdet()


def cond(a: Any, *, deps: OpsDeps, p: Any = None) -> float:
    """Condition number of a matrix, with NumPy's definition and default (F-003).

    p = None or 2: ratio of largest to smallest singular value; p = -2: the inverse
    ratio; p in {1, -1, inf, -inf, 'fro', 'nuc'}: ||A||_p * ||A^-1||_p. A singular
    matrix gives inf, as in NumPy.
    """
    np_module = deps.np_module
    if p is None or p == 2 or p == -2:
        # Singular values via NumPy (same path as svdvals; native SVD routing is F-011).
        s = np_module.linalg.svd(_to_numpy_matrix(a, deps=deps), compute_uv=False)
        if s.size == 0:
            raise ValueError("cond of an empty matrix is undefined")
        smax, smin = float(s.max()), float(s.min())
        if p == -2:
            return smin / smax if smax != 0 else float("inf")
        return smax / smin if smin != 0 else float("inf")
    try:
        inv_a = invert(a, deps=deps)
    except Exception:  # singular: NumPy returns inf here too
        return float("inf")
    return float(norm(a, p, deps=deps) * norm(inv_a, p, deps=deps))


def _engine_matrix(a: Any, *, deps: OpsDeps) -> Any:
    """A PyCauset matrix as is; anything else (NumPy array, nested lists) as a dense one."""
    if isinstance(a, deps.native.MatrixBase):
        return a
    return deps.Matrix(deps.np_module.asarray(a))


def _native_result(out: Any, *, deps: OpsDeps) -> Any:
    for item in out if isinstance(out, tuple) else (out,):
        _track_and_mark_temporary_if_native(item, deps=deps)
    return out


# Eigen decompositions, Cholesky, QR and pinv run in the engine (math/LinearAlgebra.cpp,
# compute/cpu/CpuLinalg.cpp) for every dtype, structure and view, in NumPy's LAPACK dtype;
# no NumPy fallback (F-090). eigh/eigvalsh/cholesky read the lower triangle, as NumPy does.

def eigh(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Eigenvalues (real, ascending) and eigenvectors of a symmetric / Hermitian matrix."""
    _record_io_trace("eigh", [a], deps=deps, supports_streaming=False)
    return _native_result(tuple(deps.native.eigh(_engine_matrix(a, deps=deps))), deps=deps)


def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
    """Eigenvalues (real, ascending) of a symmetric / Hermitian matrix."""
    _record_io_trace("eigvalsh", [a], deps=deps, supports_streaming=False)
    return _native_result(deps.native.eigvalsh(_engine_matrix(a, deps=deps)), deps=deps)


def eig(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Complex eigenvalues and right eigenvectors of a general square matrix."""
    _record_io_trace("eig", [a], deps=deps, supports_streaming=False)
    return _native_result(tuple(deps.native.eig(_engine_matrix(a, deps=deps))), deps=deps)


def eigvals(a: Any, *, deps: OpsDeps) -> Any:
    """Complex eigenvalues of a general square matrix."""
    _record_io_trace("eigvals", [a], deps=deps, supports_streaming=False)
    return _native_result(deps.native.eigvals(_engine_matrix(a, deps=deps)), deps=deps)


def eigvals_arnoldi(a: Any, k: int, m: int, tol: float, *, deps: OpsDeps) -> Any:
    """Top-k eigenvalues by magnitude via restarted Lanczos/Arnoldi (native, Spectra).

    Symmetric input returns real (float64) eigenvalues, otherwise complex128. The
    matrix is only touched through matrix-vector products, so disk-backed matrices
    work without materializing. Raises RuntimeError if the iteration does not converge
    to `tol` (increase `m`). `m` is the Krylov subspace size (raised to at least 2k+1).
    """
    rec = _record_io_trace("eigvals_arnoldi", [a], deps=deps)
    native_fn = getattr(deps.native, "eigvals_arnoldi", None)
    if native_fn is None:
        raise NotImplementedError("eigvals_arnoldi requires the native extension")
    out = native_fn(a, int(k), int(m), float(tol))
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


def eig_skew(a: Any, k: int, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Top-k eigenvalues + eigenvectors of a real skew-symmetric matrix.

    A real skew-symmetric matrix (A == -A.T) has purely imaginary eigenvalues in
    +/-i*lambda pairs. Returns ``(w, v)`` where ``w`` holds the top-|k| eigenvalues
    by magnitude and ``v`` is the n x min(k, n) matrix of the matching (complex)
    eigenvectors, sorted the same way. This is the diagonalization the Sorkin-Johnston
    W = positive part of i*Delta needs.
    """

    if k <= 0:
        raise ValueError("eig_skew: k must be positive")
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eig_skew requires a square matrix")

    native_fn = getattr(deps.native, "eig_skew", None)
    if callable(native_fn):
        try:
            w, v = native_fn(a, k)
            _track_and_mark_temporary_if_native(w, deps=deps)
            _track_and_mark_temporary_if_native(v, deps=deps)
            return w, v
        except Exception:
            pass

    np_module = deps.np_module
    if np_module is None:
        raise NotImplementedError("eig_skew is not available (no native/NumPy fallback)")

    w, v = np_module.linalg.eig(_to_numpy_matrix(a, deps=deps))
    idx = sorted(range(len(w)), key=lambda i: abs(w[i]), reverse=True)[:k]
    top_w = np_module.array([w[i] for i in idx])
    top_v = np_module.stack([v[:, i] for i in idx], axis=1)
    return _as_pycauset_vector(top_w, deps=deps), _as_pycauset_array(top_v, deps=deps)


def solve_triangular(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve A X = B for a triangular (or diagonal) A; raises if A is not marked triangular.

    The engine's solve handles every structure, view and dtype (F-068); the diagonal path
    this used to have dropped imaginary parts and broadcast a vector right-hand side into an
    n x n matrix (F-068).
    """
    if _effective_structure_for(a) not in ("upper_triangular", "lower_triangular", "diagonal", "identity"):
        raise ValueError("solve_triangular: a is not marked triangular")
    return solve(a, b, deps=deps)


def pinv(a: Any, *, deps: OpsDeps) -> Any:
    """Moore-Penrose pseudoinverse, as NumPy's pinv: A = U S Vᴴ → A⁺ = V S⁺ Uᴴ, singular values
    at or below max(m, n) · eps · max(s) treated as zero (F-079, F-090)."""
    _record_io_trace("pinv", [a], deps=deps)
    return _native_result(deps.native.pinv(_engine_matrix(a, deps=deps)), deps=deps)


def trace(a: Any, *, deps: OpsDeps) -> Any:
    """Sum of the diagonal: exact int for integer matrices, float, or complex (F-059)."""
    _record_io_trace("trace", [a], deps=deps)
    if isinstance(a, deps.native.MatrixBase):
        return a.trace()
    return deps.np_module.trace(_to_numpy_matrix(a, deps=deps))


def determinant(a: Any, *, deps: OpsDeps) -> Any:
    """Determinant of a square matrix: complex for complex matrices, else float (F-059)."""
    _record_io_trace("determinant", [a], deps=deps)
    if isinstance(a, deps.native.MatrixBase):
        return a.determinant()
    return deps.np_module.linalg.det(_to_numpy_matrix(a, deps=deps))


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
    """Singular values of a matrix (the S vector of an SVD), descending; engine SVD (F-079)."""
    _record_io_trace("svdvals", [a], deps=deps)
    _u, s, _vt = svd(a, full_matrices=False, deps=deps)
    return s


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

    Structural shortcut: zero stays zero (n > 0).
    """
    _record_io_trace("matrix_power", [a], deps=deps)
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("matrix_power requires a square matrix")

    if n == 0:
        return _identity_like(a, shape, deps)
    if n == 1:
        return a

    # No identity shortcut: a scaled identity (2*I)^n is 2^n*I, and identity x identity is
    # already O(1) in the engine (F-067).
    a_struct = _effective_structure_for(a)
    if a_struct == "zero":
        if n > 0:
            return a
        raise ValueError("matrix_power: zero matrix has no negative power")

    if n < 0:
        inv = invert(a, deps=deps)
        return matrix_power(inv, -n, deps=deps)

    # Binary exponentiation via matmul. No identity seed: multiplying by a float64
    # identity would promote integer/causal powers to float (and costs one extra matmul).
    result = None
    base = a
    while n > 0:
        if n & 1:
            result = base if result is None else matmul(result, base, deps=deps)
        n >>= 1
        if n:
            base = matmul(base, base, deps=deps)
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
    """Lower Cholesky factor L, A = L Lᴴ (reads the lower triangle, as NumPy). Not positive
    definite: ValueError (NumPy's LinAlgError is a ValueError)."""
    _record_io_trace("cholesky", [a], deps=deps)
    return _native_result(deps.native.cholesky(_engine_matrix(a, deps=deps)), deps=deps)


def qr(a: Any, mode: str = 'reduced', *, deps: OpsDeps) -> Any:
    """QR decomposition with NumPy's modes 'reduced' (Q m×k, R k×n), 'complete' (Q m×m, R m×n)
    and 'r' (R only). NumPy's 'raw' (LAPACK's Householder form) is not offered."""
    _record_io_trace("qr", [a], deps=deps)
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(f"qr: mode must be 'reduced', 'complete' or 'r' (got {mode!r})")
    q, r = deps.native.qr(_engine_matrix(a, deps=deps), mode == "complete")
    if mode == "r":
        return _native_result(r, deps=deps)
    return _native_result((q, r), deps=deps)


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, *, deps: OpsDeps) -> Any:
    """SVD with NumPy's signature and shapes, computed by the engine (?gesdd) for every dtype,
    structure and view; no NumPy fallback (F-079)."""
    _record_io_trace("svd", [a], deps=deps)
    if not isinstance(a, deps.native.MatrixBase):
        a = deps.Matrix(deps.np_module.asarray(a))
    u, s, vt = deps.native.svd(a, bool(full_matrices) and bool(compute_uv))
    for x in (u, s, vt):
        _track_and_mark_temporary_if_native(x, deps=deps)
    return (u, s, vt) if compute_uv else s


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
    """Solve AX = B (B a matrix or a vector) in the engine: any structure, view and dtype.

    The engine writes structured/viewed operands out and runs LAPACK ?gesv (F-068). There is
    no NumPy fallback: an engine error is the answer.
    """
    _record_io_trace("solve", [a, b], deps=deps)
    if _effective_structure_for(a) == "zero":
        raise ValueError("solve: matrix marked is_zero; system is singular")
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    a = a if isinstance(a, native) else _as_pycauset_array(deps.np_module.asarray(a), deps=deps)
    b = b if isinstance(b, native) else _as_pycauset_array(deps.np_module.asarray(b), deps=deps)
    x = deps.native.solve(a, b)
    _track_and_mark_temporary_if_native(x, deps=deps)
    return x
