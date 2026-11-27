from __future__ import annotations

import functools

import os
from typing import Any

from . import export_guard


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
    ) -> None:
        self.native = native
        self.np_module = np_module
        self.Matrix = Matrix
        self.track_matrix = track_matrix
        self.mark_temporary_if_auto = mark_temporary_if_auto
        self.warnings = warnings_module
        self.io_observer = io_observer


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


def _traced(op: str, n_operands: int = 1):
    """Write an IO trace around the call (F-100): the first `n_operands` positional
    arguments are the operands; the engine's kernel log says what ran."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args: Any, deps: OpsDeps, **kwargs: Any) -> Any:
            observer = deps.io_observer
            if observer is None:
                return fn(*args, deps=deps, **kwargs)
            with observer.trace(op, list(args[:n_operands])):
                return fn(*args, deps=deps, **kwargs)
        return inner
    return wrap


@_traced("matmul", 2)
def matmul(a: Any, b: Any, *, deps: OpsDeps) -> Any:
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
            # The engine dispatches every structure and dtype itself. The conversions that
            # were here (symmetric -> dense float64, triangular/diagonal -> float64) changed
            # result dtypes and duplicated engine paths (F-073).
            result = a @ b
            deps.track_matrix(result)
            deps.mark_temporary_if_auto(result)
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


@_traced("bitwise_and", 2)
def bitwise_and(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a & b with NumPy's dtype rules (bool & bool is bool; integers promote). Engine only."""
    return a & b


@_traced("bitwise_or", 2)
def bitwise_or(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a | b with NumPy's dtype rules. Engine only."""
    return a | b


@_traced("bitwise_xor", 2)
def bitwise_xor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a ^ b with NumPy's dtype rules. Engine only."""
    return a ^ b


@_traced("bitwise_nand", 2)
def bitwise_nand(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NAND (NOT AND)."""
    return bitwise_not(bitwise_and(a, b, deps=deps), deps=deps)


@_traced("bitwise_nor", 2)
def bitwise_nor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NOR (NOT OR)."""
    return bitwise_not(bitwise_or(a, b, deps=deps), deps=deps)


@_traced("bitwise_xnor", 2)
def bitwise_xnor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise XNOR (NOT XOR)."""
    return bitwise_not(bitwise_xor(a, b, deps=deps), deps=deps)


@_traced("invert")
def invert(matrix: Any, *, deps: OpsDeps) -> Any:
    if not isinstance(matrix, deps.native.MatrixBase):
        raise TypeError("Object does not support matrix inversion.")
    # The engine inverts every dtype, structure and view (LinearAlgebra.cpp `inverse`); a
    # singular matrix raises. No NumPy fallback.
    result = matrix.invert()
    _track_and_mark_temporary_if_native(result, deps=deps)
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


@_traced("cond")
def cond(a: Any, *, deps: OpsDeps, p: Any = None) -> float:
    """Condition number, NumPy's definition (F-003): p None/2 is smax/smin, -2 is smin/smax
    (singular values from the engine's SVD); other p is ||A||_p * ||A^-1||_p. Singular: inf."""
    a = _engine_matrix(a, deps=deps)
    if p is None or p == 2 or p == -2:
        smax = deps.native.matrix_norm(a, "2")
        smin = deps.native.matrix_norm(a, "-2")
        if p == -2:
            return smin / smax if smax != 0 else float("inf")
        return smax / smin if smin != 0 else float("inf")
    try:
        inv_a = invert(a, deps=deps)
    except RuntimeError:  # singular: NumPy returns inf here too
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

@_traced("eigh")
def eigh(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Eigenvalues (real, ascending) and eigenvectors of a symmetric / Hermitian matrix."""
    return _native_result(tuple(deps.native.eigh(_engine_matrix(a, deps=deps))), deps=deps)


@_traced("eigvalsh")
def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
    """Eigenvalues (real, ascending) of a symmetric / Hermitian matrix."""
    return _native_result(deps.native.eigvalsh(_engine_matrix(a, deps=deps)), deps=deps)


@_traced("eig")
def eig(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Complex eigenvalues and right eigenvectors of a general square matrix."""
    return _native_result(tuple(deps.native.eig(_engine_matrix(a, deps=deps))), deps=deps)


@_traced("eigvals")
def eigvals(a: Any, *, deps: OpsDeps) -> Any:
    """Complex eigenvalues of a general square matrix."""
    return _native_result(deps.native.eigvals(_engine_matrix(a, deps=deps)), deps=deps)


@_traced("eigvals_arnoldi")
def eigvals_arnoldi(a: Any, k: int, m: int, tol: float, *, deps: OpsDeps) -> Any:
    """Top-k eigenvalues by magnitude via restarted Lanczos/Arnoldi (native, Spectra).

    Symmetric input returns real (float64) eigenvalues, otherwise complex128. The
    matrix is only touched through matrix-vector products, so disk-backed matrices
    work without materializing. Raises RuntimeError if the iteration does not converge
    to `tol` (increase `m`). `m` is the Krylov subspace size (raised to at least 2k+1).
    """
    native_fn = getattr(deps.native, "eigvals_arnoldi", None)
    if native_fn is None:
        raise NotImplementedError("eigvals_arnoldi requires the native extension")
    out = native_fn(a, int(k), int(m), float(tol))
    return out


def eigvals_skew(a: Any, k: int, *, deps: OpsDeps) -> Any:
    """Top-k (by magnitude) eigenvalues of a real skew-symmetric matrix.

    A real skew-symmetric matrix (A == -A.T) has purely imaginary eigenvalues
    that come in +/-i*lambda pairs (plus a zero eigenvalue for odd dimension).
    Engine only (LAPACK on the skew-symmetric tridiagonal form); no NumPy fallback.
    """

    if k <= 0:
        raise ValueError("eigvals_skew: k must be positive")
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("eigvals_skew requires a square matrix")

    result = deps.native.eigvals_skew(_engine_matrix(a, deps=deps), k)
    _track_and_mark_temporary_if_native(result, deps=deps)
    return result


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

    return _native_result(tuple(deps.native.eig_skew(_engine_matrix(a, deps=deps), k)), deps=deps)


def solve_triangular(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve A X = B for a triangular, diagonal or identity A (by class: TriangularMatrix and
    its transpose, DiagonalMatrix, IdentityMatrix); anything else raises ValueError.

    The engine's solve handles every structure, view and dtype (F-068). It used to accept
    any matrix whose `is_*_triangular` property was set, and not a TriangularMatrix (F-092).
    """
    if not isinstance(a, deps.native.MatrixBase) or deps.native._structure_of(a) not in (
        "triangular", "diagonal", "identity"
    ):
        raise ValueError("solve_triangular: a must be a TriangularMatrix, DiagonalMatrix or IdentityMatrix")
    return solve(a, b, deps=deps)


@_traced("pinv")
def pinv(a: Any, *, deps: OpsDeps) -> Any:
    """Moore-Penrose pseudoinverse, as NumPy's pinv: A = U S Vᴴ → A⁺ = V S⁺ Uᴴ, singular values
    at or below max(m, n) · eps · max(s) treated as zero (F-079, F-090)."""
    return _native_result(deps.native.pinv(_engine_matrix(a, deps=deps)), deps=deps)


@_traced("trace")
def trace(a: Any, *, deps: OpsDeps) -> Any:
    """Sum of the diagonal: exact int for integer matrices, float, or complex (F-059)."""
    return _engine_matrix(a, deps=deps).trace()


@_traced("determinant")
def determinant(a: Any, *, deps: OpsDeps) -> Any:
    """Determinant of a square matrix: complex for complex matrices, else float (F-059)."""
    return _engine_matrix(a, deps=deps).determinant()


def _ord_name(ord: Any) -> str:
    if isinstance(ord, str):
        return ord
    if ord in (float("inf"), float("-inf")):
        return "inf" if ord > 0 else "-inf"
    if ord in (1, -1, 2, -2):
        return str(int(ord))
    raise ValueError(f"norm: invalid norm order {ord!r} for matrices")


@_traced("norm")
def norm(x: Any, ord: Any = None, *, deps: OpsDeps) -> float:
    """Matrix or vector norm with NumPy's orders, in the engine.

    Matrices: None/'fro', 'nuc', 1, -1, 2, -2, inf, -inf. Vectors: None (2-norm), any real p,
    inf, -inf, 0 (count of non-zeros).
    """
    if isinstance(x, deps.native.VectorBase):
        return float(deps.native.norm(x) if ord is None else deps.native.vector_norm(x, float(ord)))
    x = _engine_matrix(x, deps=deps)
    if ord is None or ord == "fro":
        return float(deps.native.norm(x))
    return float(deps.native.matrix_norm(x, _ord_name(ord)))


@_traced("svdvals")
def svdvals(a: Any, *, deps: OpsDeps) -> Any:
    """Singular values of a matrix (the S vector of an SVD), descending; engine SVD (F-079)."""
    _u, s, _vt = svd(a, full_matrices=False, deps=deps)
    return s


@_traced("matrix_rank")
def matrix_rank(a: Any, tol: Any = None, *, deps: OpsDeps) -> int:
    """Numerical rank: singular values above tol (NumPy's default tol when None), from the
    engine's SVD. The structural shortcuts that were here counted the non-zero diagonal of a
    triangular matrix (wrong: a strictly triangular matrix of ones has rank n - 1) and
    trusted user flags (F-090)."""
    return int(deps.native.matrix_rank(_engine_matrix(a, deps=deps), -1.0 if tol is None else float(tol)))


@_traced("matrix_power")
def matrix_power(a: Any, n: int, *, deps: OpsDeps) -> Any:
    """Integer power of a square matrix (A^n) via binary exponentiation.

    Structural shortcut: zero stays zero (n > 0).
    """
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("matrix_power requires a square matrix")

    if n == 0:
        return _identity_like(a, shape, deps)
    if n == 1:
        return a

    # No identity shortcut: a scaled identity (2*I)^n is 2^n*I, and identity x identity is
    # already O(1) in the engine (F-067).
    # No is_zero shortcut: results come from the values (F-090).
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


def _engine_vector(x: Any, *, deps: OpsDeps) -> Any:
    """A PyCauset vector as is; an array-like flattened into one (NumPy's ravel). A PyCauset
    matrix raises: outer/cross/vecdot take vectors."""
    if isinstance(x, deps.native.VectorBase):
        return x
    if isinstance(x, deps.native.MatrixBase):
        raise TypeError("expected a vector, got a matrix")
    import pycauset  # the public constructor (every dtype); the package is loaded by now

    return pycauset.vector(deps.np_module.ravel(deps.np_module.asarray(x)))


@_traced("outer", 2)
def outer(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Outer product out[i, j] = a[i] * b[j], as NumPy's outer (any lengths, multiply dtype)."""
    return _native_result(deps.native.outer(_engine_vector(a, deps=deps), _engine_vector(b, deps=deps)), deps=deps)


@_traced("cross", 2)
def cross(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Cross product of two 3-element vectors, as NumPy's cross (multiply dtype; bool raises)."""
    return _native_result(deps.native.cross(_engine_vector(a, deps=deps), _engine_vector(b, deps=deps)), deps=deps)


@_traced("vecdot", 2)
def vecdot(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Conjugate dot product sum(conj(a) * b), as NumPy's vdot; for real input this is dot."""
    return _engine_vector(a, deps=deps).conj().dot(_engine_vector(b, deps=deps))


@_traced("cholesky")
def cholesky(a: Any, *, deps: OpsDeps) -> Any:
    """Lower Cholesky factor L, A = L Lᴴ (reads the lower triangle, as NumPy). Not positive
    definite: ValueError (NumPy's LinAlgError is a ValueError)."""
    return _native_result(deps.native.cholesky(_engine_matrix(a, deps=deps)), deps=deps)


@_traced("qr")
def qr(a: Any, mode: str = 'reduced', *, deps: OpsDeps) -> Any:
    """QR decomposition with NumPy's modes 'reduced' (Q m×k, R k×n), 'complete' (Q m×m, R m×n)
    and 'r' (R only). NumPy's 'raw' (LAPACK's Householder form) is not offered."""
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(f"qr: mode must be 'reduced', 'complete' or 'r' (got {mode!r})")
    q, r = deps.native.qr(_engine_matrix(a, deps=deps), mode == "complete")
    if mode == "r":
        return _native_result(r, deps=deps)
    return _native_result((q, r), deps=deps)


@_traced("svd")
def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, *, deps: OpsDeps) -> Any:
    """SVD with NumPy's signature and shapes, computed by the engine (?gesdd) for every dtype,
    structure and view; no NumPy fallback (F-079)."""
    if not isinstance(a, deps.native.MatrixBase):
        a = deps.Matrix(deps.np_module.asarray(a))
    u, s, vt = deps.native.svd(a, bool(full_matrices) and bool(compute_uv))
    for x in (u, s, vt):
        _track_and_mark_temporary_if_native(x, deps=deps)
    return (u, s, vt) if compute_uv else s


@_traced("lu")
def lu(a: Any, *, deps: OpsDeps) -> Any:
    """(P, L, U) with A = P L U, as SciPy's lu: P real (m×m), L unit lower (m×k), U (k×n)."""
    return _native_result(tuple(deps.native.lu(_engine_matrix(a, deps=deps))), deps=deps)


@_traced("solve", 2)
def solve(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve AX = B (B a matrix or a vector) in the engine: any structure, view and dtype.

    The engine writes structured/viewed operands out and runs LAPACK ?gesv (F-068). There is
    no NumPy fallback: an engine error is the answer.
    """
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    a = a if isinstance(a, native) else _as_pycauset_array(deps.np_module.asarray(a), deps=deps)
    b = b if isinstance(b, native) else _as_pycauset_array(deps.np_module.asarray(b), deps=deps)
    x = deps.native.solve(a, b)
    _track_and_mark_temporary_if_native(x, deps=deps)
    return x
