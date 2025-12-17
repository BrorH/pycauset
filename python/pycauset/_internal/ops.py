from __future__ import annotations

from typing import Any


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
    if hasattr(matrix, "invert"):
        return matrix.invert()

    np_module = deps.np_module
    if np_module is not None:
        try:
            return np_module.linalg.inv(matrix)
        except Exception:
            pass

    raise TypeError("Object does not support matrix inversion.")
