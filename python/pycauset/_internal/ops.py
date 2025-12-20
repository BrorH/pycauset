from __future__ import annotations

from typing import Any


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


def _to_numpy_matrix(obj: Any, *, deps: OpsDeps) -> Any:
    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for this operation")
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
    ) -> None:
        self.native = native
        self.np_module = np_module
        self.Matrix = Matrix
        self.TriangularBitMatrix = TriangularBitMatrix
        self.track_matrix = track_matrix
        self.mark_temporary_if_auto = mark_temporary_if_auto
        self.warnings = warnings_module


def matmul(a: Any, b: Any, *, deps: OpsDeps) -> Any:
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

    if native_matmul is not None and native_matrix_base is not None:
        if isinstance(a, native_matrix_base) and isinstance(b, native_matrix_base):
            result = native_matmul(a, b)
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)
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
            return deps.Matrix(res_np)
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
    return deps.Matrix(res_data)


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
    native_exc: Exception | None = None
    if hasattr(matrix, "invert"):
        try:
            result = matrix.invert()
        except Exception as exc:
            native_exc = exc
        else:
            _track_and_mark_temporary_if_native(result, deps=deps)
            return result

    np_module = deps.np_module
    if np_module is not None:
        try:
            result = np_module.linalg.inv(matrix)
            return _as_pycauset_array(result, deps=deps)
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
    """Eigen-decomposition for real symmetric / complex Hermitian matrices (NumPy fallback)."""
    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for eigh")
    w, v = np_module.linalg.eigh(_to_numpy_matrix(a, deps=deps))
    w_out = _as_pycauset_array(w, deps=deps)
    v_out = _as_pycauset_array(v, deps=deps)
    return w_out, v_out


def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
    """Eigenvalues for real symmetric / complex Hermitian matrices (NumPy fallback)."""
    np_module = deps.np_module
    if np_module is None:
        raise RuntimeError("NumPy is required for eigvalsh")
    w = np_module.linalg.eigvalsh(_to_numpy_matrix(a, deps=deps))
    return _as_pycauset_array(w, deps=deps)


def solve_triangular(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError(
        "solve_triangular is not available yet; it will be added once triangular flags/solvers land"
    )


def lu(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("lu is not available yet")


def cholesky(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("cholesky is not available yet")


def svd(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("svd is not available yet")


def pinv(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("pinv is not available yet")
