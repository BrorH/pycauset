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

    if isinstance(a, deps.TriangularBitMatrix) and isinstance(b, deps.TriangularBitMatrix):
        if native_matmul is None:
            raise NotImplementedError("Native matmul is not available in this build.")
        result = native_matmul(a, b)
        deps.track_matrix(result)
        deps.mark_temporary_if_auto(result)
        return result

    # Generic fallback
    if not (hasattr(a, "shape") and hasattr(b, "shape")):
        raise TypeError("Inputs must be matrix-like objects with a shape property.")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    N = a.shape[0]
    M = b.shape[1]

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

    # Slow generic loop
    res = deps.Matrix(N)
    if N != M:
        raise NotImplementedError(
            "Generic multiplication currently only supports square result matrices."
        )

    for i in range(N):
        for j in range(M):
            val = 0
            for k in range(a.shape[1]):
                val += a.get(i, k) * b.get(k, j)
            res.set(i, j, val)

    return res


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
